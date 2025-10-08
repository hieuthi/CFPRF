import time
import torch,argparse
import numpy as np
from libs.startup_config import set_random_seed
from models.FDN import * 
import torchaudio



def get_basename(filepath):
    path, fname = os.path.split(filepath)
    bname, ext  = os.path.splitext(fname)
    return bname

def get_truncated_segment(segment, size, pad_value=0):
  if segment.size(-1) < size:
    segment = F.pad(segment, (0, size - segment.size(-1)), value=pad_value)
  else:
    segment = segment[..., :size]
  return segment

def load_padded_waveforms(filepaths):
    trailing_pad = int(0.005*16000)
    target_size  = 0
    waveforms = []
    Ts        = []
    for filepath in filepaths:
        waveform, sample_rate = torchaudio.load(filepath)
        size = waveform.shape[-1]
        target_size = target_size if target_size > 0 else max(size + trailing_pad, int(0.645*16000))
        waveform = get_truncated_segment(waveform, target_size)
        waveforms.append(torch.unsqueeze(waveform,0))
        Ts.append(size)
    return torch.cat(waveforms), Ts

def produce_evaluation(scp, model, outpath, batch_size=2, device='cuda'):
    st  = time.time()

    utterances = []
    with open(scp, 'r') as f:
        for line in f:
            path = line.strip()
            size = os.path.getsize(path)
            name = get_basename(path)
            utterances.append({"basename": name, "path": path, "size": size})
    utterances.sort(key=lambda x: x["size"], reverse=True)

    model.to(device)

    startIdx = 0
    model.eval()
    with torch.inference_mode():
      while startIdx < len(utterances):
        endIdx = min(startIdx+batch_size,len(utterances))
        B      = endIdx - startIdx
        filepaths = [ utterance["path"] for utterance in utterances[startIdx:endIdx] ]

        waveforms, Ts = load_padded_waveforms(filepaths)
        waveforms = waveforms.to(device)
        seg_score,_, embs, last_hidden_states = model(torch.squeeze(waveforms,1)) 


        scalepath = f"{outpath}/unit0.02.score"
        scores = torch.reshape(seg_score, (B, -1, 2)).cpu().numpy()
        with open(scalepath, 'a+') as f:
          for idx in range(B):
            utterance = utterances[startIdx + idx]
            name      = utterance["basename"]
            T    = Ts[idx] 
            nsegs = max(int(T/(0.02*16000)),1)
            score = scores[idx,:nsegs, :]
            f.write("\n".join(f"{name} {i} {item[0]:.05f} {item[1]:.05f}" for i, item in enumerate(score))+"\n") 


          startIdx = endIdx
        duration  = time.time() - st
    print(f"FINISHED: Inference took {duration/60:.02f} min")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('python evaluate_CFPRF.py --save_path result_ps --eval_scp PartialSpoof/scp/eval.scp')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--save_path', type=str, default="./result/") # ['HAD','PS','LAVDF']
    parser.add_argument('--seql', type=int, default=1070)
    parser.add_argument('--glayer', type=int, default=1) 
    parser.add_argument("--scp", default=None, type=str)
    parser.add_argument("--resume_ckpt", type=str, default=None)
    args = parser.parse_args()

    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FDN_model = CFPRF_FDN(seq_len=args.seql, gmlp_layers=args.glayer).to(device)
    FDN_model.load_state_dict(torch.load(args.resume_ckpt))

    """makedir"""
    os.makedirs(args.save_path,exist_ok=True)
    produce_evaluation(args.scp, FDN_model, args.save_path, batch_size=2, device=device)
