import glob
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch

from inference.infer_tool import Svc

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('markdown_it').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('multipart').setLevel(logging.WARNING)

model = None
spk = None
debug = False

local_model_root = './trained'

cuda = {}
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_properties(i).name
        cuda[f"CUDA:{i} {device_name}"] = f"cuda:{i}"



def modelAnalysis(model_path,config_path,cluster_model_path,device,enhance,diff_model_path,diff_config_path,only_diffusion,use_spk_mix,local_model_enabled,local_model_selection):
    global model
    device = cuda[device] if "CUDA" in device else device
    cluster_filepath = os.path.split(cluster_model_path.name) if cluster_model_path is not None else "no_cluster"
    # get model and config path
    if (local_model_enabled):
        # local path
        model_path = glob.glob(os.path.join(local_model_selection, '*.pth'))[0]
        config_path = glob.glob(os.path.join(local_model_selection, '*.json'))[0]
    else:
        # upload from webpage
        model_path = model_path.name
        config_path = config_path.name
    fr = ".pkl" in cluster_filepath[1]
    model = Svc(model_path,
            config_path,
            device=device if device != "Auto" else None,
            cluster_model_path = cluster_model_path.name if cluster_model_path is not None else "",
            nsf_hifigan_enhance=enhance,
            diffusion_model_path = diff_model_path.name if diff_model_path is not None else "",
            diffusion_config_path = diff_config_path.name if diff_config_path is not None else "",
            shallow_diffusion = True if diff_model_path is not None else False,
            only_diffusion = only_diffusion,
            spk_mix_enable = use_spk_mix,
            feature_retrieval = fr
            )
    spks = list(model.spk2id.keys())
    device_name = torch.cuda.get_device_properties(model.dev).name if "cuda" in str(model.dev) else str(model.dev)
    msg = f"成功加载模型到设备{device_name}上\n"
    if cluster_model_path is None:
        msg += "未加载聚类模型或特征检索模型\n"
    elif fr:
        msg += f"特征检索模型{cluster_filepath[1]}加载成功\n"
    else:
        msg += f"聚类模型{cluster_filepath[1]}加载成功\n"
    if diff_model_path is None:
        msg += "未加载扩散模型\n"
    else:
        msg += f"扩散模型{diff_model_path.name}加载成功\n"
    msg += "当前模型的可用音色：\n"
    for i in spks:
        msg += i + " "
    print(msg)


def modelUnload():
    global model
    if model:
        model.unload_model()
        model = None
        torch.cuda.empty_cache()
        print("模型卸载完毕!")
    print("模型已经卸载或者未加载")
    
def vc_infer(output_format, sid, audio_path, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment):
    global model
    _audio = model.slice_inference(
        audio_path,
        sid,
        vc_transform,
        slice_db,
        cluster_ratio,
        auto_f0,
        noise_scale,
        pad_seconds,
        cl_num,
        lg_num,
        lgr_num,
        f0_predictor,
        enhancer_adaptive_key,
        cr_threshold,
        k_step,
        use_spk_mix,
        second_encoding,
        loudness_envelope_adjustment
    )  
    model.clear_empty()
    #构建保存文件的路径，并保存到results文件夹内
    str(int(time.time()))
    if not os.path.exists("results"):
        os.makedirs("results")
    key = "auto" if auto_f0 else f"{int(vc_transform)}key"
    cluster = "_" if cluster_ratio == 0 else f"_{cluster_ratio}_"
    isdiffusion = "sovits"
    if model.shallow_diffusion:
        isdiffusion = "sovdiff"

    if model.only_diffusion:
        isdiffusion = "diff"
    
    output_file_name = 'result_'+truncated_basename+f'_{sid}_{key}{cluster}{isdiffusion}.{output_format}'
    output_file = os.path.join("results", output_file_name)
    soundfile.write(output_file, _audio, model.target_sample, format=output_format)
    return output_file

def vc_fn(sid, input_audio, output_format, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold,k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment):
    global model
    if input_audio is None:
        return "You need to upload an audio", None
    if model is None:
        return "You need to upload an model", None
    if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
        if cluster_ratio != 0:
            return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None
    #print(input_audio)    
    audio, sampling_rate = soundfile.read(input_audio)
    #print(audio.shape,sampling_rate)
    if np.issubdtype(audio.dtype, np.integer):
        audio = (audio / np.iinfo(audio.dtype).max).astype(np.float32)
    #print(audio.dtype)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio.transpose(1, 0))
    # 未知原因Gradio上传的filepath会有一个奇怪的固定后缀，这里去掉
    truncated_basename = Path(input_audio).stem #[:-6]
    print("truncated_basename", truncated_basename)
    processed_audio = os.path.join("raw", f"{truncated_basename}.wav")
    print("truncated_basename", processed_audio)
    soundfile.write(processed_audio, audio, sampling_rate, format="wav")
    output_file = vc_infer(output_format, sid, processed_audio, truncated_basename, vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)

    return "Success", output_file

def vc_fn2(_text, _lang, _gender, _rate, _volume, sid, output_format, vc_transform, auto_f0,cluster_ratio, slice_db, noise_scale,pad_seconds,cl_num,lg_num,lgr_num,f0_predictor,enhancer_adaptive_key,cr_threshold, k_step,use_spk_mix,second_encoding,loudness_envelope_adjustment):
    global model
    if model is None:
        return "You need to upload an model", None
    if getattr(model, 'cluster_model', None) is None and model.feature_retrieval is False:
        if cluster_ratio != 0:
            return "You need to upload an cluster model or feature retrieval model before assigning cluster ratio!", None
    _rate = f"+{int(_rate*100)}%" if _rate >= 0 else f"{int(_rate*100)}%"
    _volume = f"+{int(_volume*100)}%" if _volume >= 0 else f"{int(_volume*100)}%"
    if _lang == "Auto":
        _gender = "Male" if _gender == "男" else "Female"
        subprocess.run([sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume, _gender])
    else:
        subprocess.run([sys.executable, "edgetts/tts.py", _text, _lang, _rate, _volume])
    target_sr = 44100
    y, sr = librosa.load("tts.wav")
    resampled_y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    soundfile.write("tts.wav", resampled_y, target_sr, subtype = "PCM_16")
    input_audio = "tts.wav"
    #audio, _ = soundfile.read(input_audio)
    output_file_path = vc_infer(output_format, sid, input_audio, "tts", vc_transform, auto_f0, cluster_ratio, slice_db, noise_scale, pad_seconds, cl_num, lg_num, lgr_num, f0_predictor, enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix, second_encoding, loudness_envelope_adjustment)
    # os.remove("tts.wav")
    return "Success", output_file_path





model_path = "./trained/NAI/output.pth"
config_path = "./trained/NAI/config.json"
cluster_model_path = None
device = "Auto"
enhance = False
diff_model_path = None
diff_config_path = None
only_diffusion = False
use_spk_mix = False
local_model_enabled = True
local_model_selection = "./trained/NAI"

modelAnalysis(
    model_path,
    config_path,
    cluster_model_path,
    device,
    enhance,
    diff_model_path,
    diff_config_path,
    only_diffusion,
    use_spk_mix,
    local_model_enabled,
    local_model_selection
    )

print("success")
print(model)


_text = "你好，中国"
_lang = "Auto"
_gender = "女"
_rate = 0
_volume = 0
sid = "nai"
output_format = "wav"
vc_transform = 0
auto_f0 = True     # 自动f0预测，配合聚类模型f0预测效果更好,会导致变调功能失效（仅限转换语音，歌声勾选此项会究极跑调）
cluster_ratio = 0
slice_db = -40
noise_scale = 0.4
pad_seconds = 0.5
cl_num = 0
lg_num = 0
lgr_num = 0.75
# f0_predictor = "pm"
f0_predictor = "crepe"
enhancer_adaptive_key = 0
cr_threshold = 0.05
k_step = 100
second_encoding = False
loudness_envelope_adjustment = 0


result = vc_fn2(
    _text,
    _lang,
    _gender,
    _rate,
    _volume,
    sid,
    output_format,
    vc_transform,
    auto_f0,
    cluster_ratio,
    slice_db,
    noise_scale,
    pad_seconds,
    cl_num,
    lg_num,
    lgr_num,
    f0_predictor,
    enhancer_adaptive_key,
    cr_threshold,
    k_step,
    use_spk_mix,
    second_encoding,
    loudness_envelope_adjustment
    )

print(result)
