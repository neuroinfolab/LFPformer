import importlib.util
from pathlib import Path
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from lfpformer.paths import REFERENCE, activate_reference
from lfpformer.geometry import np1_geometry


@pytest.mark.parametrize('kind', ['transformer', 'fusion-linear', 'fusion-transformer'])
def test_specialized_inference_matches_original_forward(tmp_path, monkeypatch, kind):
    torch.set_num_threads(2)
    activate_reference('benchmark')
    monkeypatch.syspath_prepend(str(REFERENCE/'fusion'))
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    from predict_sequence import predict
    args = dict(d_model=32,nhead=4,layers=1,ff=64,dropout=0.,gaussian=True,
                decompose=False,max_len=8,arm='lfp_manip',arch='transformer')
    x = np.random.default_rng(7).normal(size=(4,16)).astype(np.float32)
    g = np1_geometry(np.arange(4))
    m = np.tile([4000.,3000.,7000.], (4,1)).astype(np.float32)
    xt = torch.from_numpy(x)[None]
    gt = torch.tensor(g[:,[0,2]],dtype=torch.float32)[None]
    mask = torch.zeros((1,4),dtype=torch.bool)
    if kind == 'transformer':
        from train_probe_sequence_ext import ProbeSequenceModel
        model = ProbeSequenceModel(16,32,4,1,64,0.,False,True).eval()
        torch.nn.init.normal_(model.resid_head.weight, std=.01)
        ck = dict(model=model.state_dict(),args=args)
        with torch.no_grad():
            mu,sigma=model(xt,gt,gt[...,1],mask)
        expected=(mu[0].numpy(),sigma[0].numpy())
    elif kind == 'fusion-transformer':
        import experiment as e
        monkeypatch.setattr(e,'CONFIG',{**e.CONFIG,**args})
        model=e.MatchedDirectProbeSeq(16,np.ones(16)*.1,2.,'lfp_manip').eval()
        torch.nn.init.normal_(model.resid_head.weight,std=.01)
        ck=dict(model=model.state_dict(),args=args)
        with torch.no_grad():
            mu,sigma=model(xt,torch.from_numpy(m)[None],gt,gt[...,1],mask)
        expected=(mu[0].numpy(),sigma[0].numpy())
    else:
        from eval_manip_ft import make_heads, normalizer_from_stats
        _,model=make_heads('linear',19,'cpu');model.eval()
        stats=dict(ap_min=0.,ap_max=10000.,dv_min=0.,dv_max=7000.,d_min=0.,d_max=5739.,midline=5739.)
        nz=normalizer_from_stats(stats)
        ck=dict(manip_conditioned=True,gauss_head_state_dict=model.state_dict(),
                normalizer_stats=stats,head_type='linear',parameterization='direct')
        with torch.no_grad():
            mu,lv=model(torch.tensor(np.concatenate([x,nz.encode(m)],axis=1)))
        expected=(nz.decode(mu.numpy()),np.exp(.5*lv.numpy())*np.array([10000,7000,5739]))
    torch.save(ck,tmp_path/'model.pt')
    data=dict(emb=x,geometry_xyz=g,probe_ids=np.repeat('p',4),channel_ids=np.arange(4),manip_xyz=m)
    for labelled in (False,True):
        if labelled:
            data['true_xyz']=np.full((4,3),99999.)
        np.savez(tmp_path/'input.npz',**data)
        predict(tmp_path/'input.npz',tmp_path/'model.pt',tmp_path/'output.npz',kind)
        with np.load(tmp_path/'output.npz') as z:
            np.testing.assert_allclose(z['mu'],expected[0],rtol=1e-5,atol=1e-3)
            np.testing.assert_allclose(z['sigma'],expected[1],rtol=1e-5,atol=1e-3)
    if kind != 'fusion-linear':
        ck['args']['max_len']=3
        torch.save(ck,tmp_path/'small.pt')
        with pytest.raises(ValueError,match='max_len'):
            predict(tmp_path/'input.npz',tmp_path/'small.pt',tmp_path/'output.npz',kind)
