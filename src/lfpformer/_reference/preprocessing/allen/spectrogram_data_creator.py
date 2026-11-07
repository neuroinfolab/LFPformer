"""
This script is designed to process Local Field Potential (LFP) data from the Allen Brain Atlas
to generate spectrograms. It contains two main functions:

1.  `create_spectrogram_mini`: This function creates spectrograms for specific, predefined brain
    regions (e.g., hippocampus and visual cortex). It labels the data by region, making it
    suitable for supervised machine learning tasks like region classification. It saves both
    the spectrograms and the raw LFP chunks.

2.  `create_spectrogram`: This function processes LFP data from all available channels for
    given sessions. It saves the raw LFP chunks along with metadata about the brain region
    for each channel, which can be used for unsupervised models like wav2vec 2.0.

The script handles data fetching, session and probe iteration, channel selection, data chunking,
spectrogram computation, and saving the results to disk.
"""
import os
import sys
import json
import warnings
import gc
import pickle
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# Add parent and grandparent directories to the system path to allow for package imports
# This is necessary for the script to find the `script.utils` module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from script.utils import compute_spectrogram_librosa

# Suppress warnings for cleaner output during execution
warnings.filterwarnings('ignore')
def _already_done(path: str, min_bytes: int = 5_000_000) -> bool:
    """
    Consider a probe done if output exists and is at least min_bytes.
    Tune min_bytes based on typical pickle size on your run.
    """
    try:
        return os.path.exists(path) and os.path.getsize(path) >= min_bytes
    except OSError:
        return False


def _atomic_pickle_dump(obj, out_path: str):
    """
    Atomic write: write to tmp then rename. Prevents partial file on time-limit kill.
    """
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, out_path)


def create_spectrogram_mini(chunk_size, start_time, n_chunks, data="Allen"):
    """
    Generates labeled spectrograms and raw LFP chunks for specified brain regions.

    This function is tailored for creating a dataset for supervised learning. It filters
    for sessions containing specific hippocampal and visual cortex regions, extracts LFP
    data, chunks it, computes spectrograms, and saves the labeled data.

    Args:
        chunk_size (int): The duration of each LFP chunk in seconds.
        start_time (int): The timepoint in seconds to start extracting data from.
        n_chunks (int): The number of chunks to extract per channel.
        data (str, optional): Identifier for the dataset. Defaults to "Allen".
    """
    print("Job start: Creating mini spectrogram dataset.")
    print(f"Current working directory: {os.getcwd()}")
    flag = True  # Flag to print spectrogram shape only once

    # --- Setup Paths and Region Definitions ---
    output_path = os.path.join("spectrogram", data)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    hc_acronyms = {'CA1', 'CA2', 'CA3', 'DG', 'Visual_Cortex'}
    acronyms_arr = np.array(sorted(list(hc_acronyms)))
    acr_dict = {acr: i for i, acr in enumerate(acronyms_arr)}
    print(f"Region to ID mapping: {acr_dict}")

    if data == "Allen":
        # NOTE: The script changes the current working directory to find the Allen SDK manifest.
        # This might cause issues if run in different contexts.
        os.chdir(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/AlphaBrain/Alphabrain_staging/data/Allen'))
        output_path = f'../../script/spectrogram/{data}'
        manifest_path = "manifest.json"

        # Initialize Allen SDK Cache
        data_cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

        # --- Session and Probe Filtering ---
        sessions_list = data_cache.get_session_table()
        mask = sessions_list['ecephys_structure_acronyms'].apply(
            lambda acronyms: bool(hc_acronyms.intersection(set(acronyms)))
        )
        filtered_sessions = sessions_list[mask]
        print(f"Total number of sessions with target regions: {len(filtered_sessions)}")

        # Discover LFP files for the filtered sessions
        probe_files_map = defaultdict(list)
        base = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/AlphaBrain/Alphabrain_staging/data/Allen/sessions')
        for root, dirs, files in os.walk(base):
            if os.path.basename(root).startswith('session_'):
                session_folder_name = os.path.basename(root).replace('session_', '')
                if int(session_folder_name) in filtered_sessions.index:
                    for file in files:
                        if file.startswith('probe_') and file.endswith('_lfp.nwb'):
                            probe_id = file.replace('_lfp.nwb', '').replace('probe_', '')
                            probe_files_map[session_folder_name].append(probe_id)
        print(f"Found probe files for processing: {probe_files_map}")

        # --- Data Processing Loop ---
        for session_id, probe_files in probe_files_map.items():
            region_counts = [0] * len(hc_acronyms)
            data_list = []  # To store spectrograms
            raw_list = []   # To store raw LFP chunks
            print(f"Processing Session: {session_id}")
            session_data = data_cache.get_session_data(int(session_id))

            for probe_id in tqdm(probe_files, desc=f"Probes for session {session_id}"):
                try:
                    lfp_data = session_data.get_lfp(int(probe_id))
                except Exception as e:
                    print(f"Could not load LFP data for probe {probe_id}: {e}")
                    continue

                sampling_rate = session_data.probes.loc[int(probe_id)].lfp_sampling_rate

                # Process hippocampal regions
                for region in hc_acronyms - {'Visual_Cortex'}:
                    chan_ids = session_data.channels[
                        (session_data.channels.probe_id == int(probe_id)) &
                        (session_data.channels.ecephys_structure_acronym == region)
                    ].index.values
                    if len(chan_ids) == 0:
                        continue

                    regional_lfp_data = lfp_data.sel(channel=slice(np.min(chan_ids), np.max(chan_ids)))
                    lfp_channel_index = regional_lfp_data.channel

                    for i, channel_data in enumerate(regional_lfp_data.data.T):
                        window_size = int(chunk_size * sampling_rate)
                        start = int(start_time * sampling_rate)
                        end = start + window_size
                        count = 0
                        while count < n_chunks and end < len(channel_data):
                            row = channel_data[start:end]
                            if not np.any(np.isnan(row)) and not np.any(np.isinf(row)):
                                z_spect = compute_spectrogram_librosa(row, sampling_rate, 500, 21)
                                y = acr_dict[region]
                                if flag:
                                    print(f"Spectrogram shape: {z_spect.shape}")
                                    flag = False

                                if not np.any(np.isnan(z_spect)) and not np.any(np.isinf(z_spect)):
                                    channel_id_val = lfp_channel_index[i].item()
                                    data_list.append((z_spect, y, count, f"{probe_id}_{channel_id_val}"))
                                    raw_list.append((row, y, count, f"{probe_id}_{channel_id_val}"))
                                    count += 1
                                    region_counts[y] += 1
                            start += window_size
                            end += window_size

                # Process Visual Cortex data only if hippocampal data exists
                if len(data_list) > 0:
                    vis_acr = {'VIS', 'VISal', 'VISam', 'VISl', 'VISli', 'VISmma', 'VISmmp', 'VISp', 'VISpm', 'VISrl'}
                    chan_ids = session_data.channels[
                        (session_data.channels.probe_id == int(probe_id)) &
                        (session_data.channels.ecephys_structure_acronym.isin(vis_acr))
                    ].index.values
                    if len(chan_ids) == 0:
                        continue
                    
                    regional_lfp_data = lfp_data.sel(channel=slice(np.min(chan_ids), np.max(chan_ids)))
                    lfp_channel_index = regional_lfp_data.channel

                    for i, channel_data in enumerate(regional_lfp_data.data.T):
                        window_size = int(chunk_size * sampling_rate)
                        start = int(start_time * sampling_rate)
                        end = start + window_size
                        count = 0
                        while count < n_chunks and end < len(channel_data):
                            row = channel_data[start:end]
                            if not np.any(np.isnan(row)) and not np.any(np.isinf(row)):
                                z_spect = compute_spectrogram_librosa(row, sampling_rate, 500, 21)
                                y = acr_dict['Visual_Cortex']
                                if not np.any(np.isnan(z_spect)) and not np.any(np.isinf(z_spect)):
                                    channel_id_val = lfp_channel_index[i].item()
                                    data_list.append((z_spect, y, count, f"{probe_id}_{channel_id_val}"))
                                    raw_list.append((row, y, count, f"{probe_id}_{channel_id_val}"))
                                    count += 1
                                    region_counts[y] += 1
                            start += window_size
                            end += window_size
                
                del lfp_data
                gc.collect()

            # --- Save Data for Session ---
            if data_list:
                with open(os.path.join(output_path, f"{session_id}_data.pickle"), 'wb') as file:
                    print(f"Saving {len(data_list)} spectrograms for session {session_id}.")
                    pickle.dump(data_list, file)
                with open(os.path.join(output_path, f"{session_id}_raw.pickle"), 'wb') as file:
                    print(f"Saving {len(raw_list)} raw LFP chunks for session {session_id}.")
                    pickle.dump(raw_list, file)
                print(f"Region counts for session {session_id}: {region_counts}")
            else:
                print(f"No data to save for session: {session_id}")
            
            del session_data
            gc.collect()


def create_spectrogram(chunk_size):
    """
    Generates raw LFP chunks and metadata for all channels in all sessions.

    This function is designed to prepare data for unsupervised learning models like
    wav2vec 2.0. It iterates through all sessions and probes, extracts LFP data for
    all channels, chunks it, and saves the raw chunks along with metadata about
    the session, probe, channel, and brain region.

    Args:
        chunk_size (int): The duration of each LFP chunk in seconds.
    """
    print("Job start: Creating full spectrogram dataset for w2v2.")
    print(f"Current working directory: {os.getcwd()}")

    # --- Setup Paths ---
    data = "Allen"
    # Note: This path seems intended for a different project structure (wav2vec2)
    output_path = os.path.join("..", "Allen_w2v2", data)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    

    # NOTE: The script changes the current working directory to find the Allen SDK manifest.
    os.chdir(__import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/AlphaBrain/Alphabrain_staging/data/Allen'))
    output_path = os.path.abspath(output_path)
    # This output path is for metadata
    metadata_output_path = f'../w2v2_metadata/{data}'
    if not os.path.exists(metadata_output_path):
        os.makedirs(metadata_output_path)
    metadata_output_path = os.path.abspath(metadata_output_path)

    manifest_path = "manifest.json"
    data_cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

    # --- Session and Probe Processing ---
    sessions_list = data_cache.get_session_table()
    print(f"Total number of sessions: {len(sessions_list)}")
    channel_metadata = data_cache.get_channels()

    # Discover all LFP files
    probe_files_map = defaultdict(list)
    base = __import__('lfpformer.paths', fromlist=['legacy_path']).legacy_path('/workspace/ws1/AlphaBrain/Alphabrain_staging/data/Allen/sessions')
    for root, dirs, files in os.walk(base):
        if os.path.basename(root).startswith('session_'):
            session_folder_name = os.path.basename(root).replace('session_', '')
            for file in files:
                if file.startswith('probe_') and file.endswith('_lfp.nwb'):
                    probe_id = file.replace('_lfp.nwb', '').replace('probe_', '')
                    probe_files_map[session_folder_name].append(probe_id)
    print(f"Found probe files for processing: {probe_files_map}")
    # Deduplicate probe IDs per session (your discovery can add duplicates)
    for sid in list(probe_files_map.keys()):
        probe_files_map[sid] = sorted(set(probe_files_map[sid]))
    session_metadata = {}
    for session_id, probe_files in probe_files_map.items():
        print(f"Processing Session: {session_id}")
        session_data = data_cache.get_session_data(int(session_id))
        probes_data = {}

        for probe_id in tqdm(probe_files, desc=f"Probes for session {session_id}"):
            out_file = os.path.join(output_path, f"{session_id}_{probe_id}.pickle")

            # Skip if already computed
            if _already_done(out_file):
                print(f"[SKIP] {session_id} probe {probe_id} already exists: {out_file}")
                continue
            probe_dump = []
            probe_metadata = {}
            try:
                lfp_data = session_data.get_lfp(int(probe_id))
            except Exception as e:
                print(f"Could not load LFP data for probe {probe_id}: {e}")
                continue

            sampling_rate = 1250  # Using a fixed sampling rate

            # Get all channel IDs for the probe
            chan_ids = session_data.channels[session_data.channels.probe_id == int(probe_id)].index.values
            if len(chan_ids) == 0:
                continue

            regional_lfp_data = lfp_data.sel(channel=slice(np.min(chan_ids), np.max(chan_ids)))
            lfp_channel_index = regional_lfp_data.channel

            for i, channel_data in enumerate(regional_lfp_data.data.T):
                window_size = int(chunk_size * sampling_rate)
                start = 0
                end = start + window_size
                count = 0
                while end < len(channel_data):
                    row = channel_data[start:end]
                    if not np.any(np.isnan(row)) and not np.any(np.isinf(row)):
                        try:
                            channel_id_val = lfp_channel_index[i].item()
                            brain_region = channel_metadata.loc[channel_id_val, 'ecephys_structure_acronym']
                        except KeyError:
                            brain_region = "unknown"
                        
                        # Create a unique ID for each chunk
                        chunk_id = f"{session_id}_{count}_{probe_id}_{channel_id_val}_{brain_region}"
                        probe_dump.append((row, chunk_id))
                        probe_metadata[brain_region] = probe_metadata.get(brain_region, 0) + 1

                    count += 1
                    start += window_size
                    end += window_size
            
            print(f"Probe {probe_id} metadata: {probe_metadata}")
            probes_data[probe_id] = probe_metadata
            
            # Save the raw LFP chunks for the current probe
            # with open(os.path.join(output_path, f"{session_id}_{probe_id}.pickle"), 'wb') as file:
            #     pickle.dump(probe_dump, file)
            _atomic_pickle_dump(probe_dump, out_file)
            del lfp_data
            gc.collect()

        session_metadata[session_id] = probes_data
        del session_data
        gc.collect()

    # Save the aggregated metadata for all processed sessions
    with open(os.path.join(metadata_output_path, "session_metadata.json"), 'w') as file:
        json.dump(session_metadata, file, indent=4)
    print("Metadata saved.")


if __name__ == '__main__':
    # This block allows the script to be run from the command line.
    # The user can choose which function to run and with which parameters.
    
    print("Running spectrogram creation script.")
    
    # --- Example for creating labeled spectrograms for classification ---
    # This will process data in 3-second chunks, starting at 200 seconds, for 100 chunks.
    # print("Running 'create_spectrogram_mini'...")
    # create_spectrogram_mini(chunk_size=3, start_time=200, n_chunks=100)

    # --- Example for creating a dataset for unsupervised learning (wav2vec 2.0) ---
    # This will process data in 3-second chunks from the beginning of the recording.
    print("Running 'create_spectrogram'...")
    create_spectrogram(chunk_size=3)
    
    print("Script finished.")
