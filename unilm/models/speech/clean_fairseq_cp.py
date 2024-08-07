import os, sys
import torch
from WavLM import WavLMConfig

cp_path = sys.argv[1]

cp = torch.load(cp_path, 'cpu')

# only keep cfg and model
cp_keys = list(cp.keys())
for cp_key in cp_keys:
    if cp_key not in ['cfg', 'model']:
        print('pop: {} of {}'.format(cp_key, 'cp'))
        cp.pop(cp_key)
print('\n')

# clean model cfg
cp['cfg']['model']['normalize'] = cp['cfg']['task']['normalize']
cp['cfg'] = cp['cfg']['model']
cfg = WavLMConfig()
cfg_keys = list(cp['cfg'])
for cfg_key in cfg_keys:
    if cfg_key not in cfg.__dict__.keys():
        print('pop: {} of {}, which is {}'.format(cfg_key, 'cp_cfg', cp['cfg'][cfg_key]))
        cp['cfg'].pop(cfg_key)
print('\n')
for cfg_key in cfg.__dict__.keys():
    if cfg_key not in cp['cfg']:
        print(f'{cfg_key} not found in cp')
print('\n')

# cp['model'].pop('label_embs_concat')
# cp['model'].pop('final_proj.weight')
# cp['model'].pop('final_proj.bias')

# self.target_glu = None
# if self.utterance_contrastive_loss:
#     self.quantizer = None
#     self.project_q = None
#     self.spk_proj = None
#     self.utterance_contrastive_loss = False
#     self.utterance_contrastive_layer = None
# if hasattr(self.encoder, "layer_norm_for_extract"):
#     self.encoder.layer_norm_for_extract = None

# clean model keys
model_keys = list(cp['model'].keys())
for key in model_keys:
    if 'label_embs_concat' in key or 'final_proj' in key or 'target_glu' in key or 'quantizer' in key \
            or 'project_q' in key or 'spk_proj' in key or 'layer_norm_for_extract' in key \
            or 'quant_in_proj' in key or 'quant_embedding' in key:
        cp['model'].pop(key)
        print(f'delete {key}')

    if 'gate_ur_linear' in key:
        new_key = key.replace('gate_ur_linear', 'grep_linear')
        cp['model'][new_key] = cp['model'][key]
        cp['model'].pop(key)
        print(f'replace {key} wih {new_key}')
    if 'eco_a' in key:
        new_key = key.replace('eco_a', 'grep_a')
        cp['model'][new_key] = cp['model'][key]
        cp['model'].pop(key)
        print(f'replace {key} wih {new_key}')

torch.save(cp, cp_path+'.for_inference')


