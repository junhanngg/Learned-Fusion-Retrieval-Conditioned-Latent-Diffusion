import numpy as np
import librosa
from IPython.display import Audio, display
import yaml
from typing import Dict

# loads in yaml file and defines the Mel2Waveform class for converting mel spectrograms back to waveforms using Griffin-Lim algorithm
class Mel2Waveform:
    """
    Convert mel spectrograms back to approximated waveforms using the Griffin-Lim algorithm.
        - Uses data_config.yaml within data/config to read in spectrogram parameters for accurate inversion.
    """
    def __init__(self, config_path, spec_scale = "log1p_power", griffinlim_iters = 32):
        # loads in config parameters from data_config which was used to create spectrograms
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        self.cfg = cfg
        self.spec_scale = spec_scale
        self.griffinlim_iters = griffinlim_iters

        self.sample_rate = cfg["audio"]["sample_rate"]

        spec_cfg = cfg["spectrogram"]
        self.n_fft = spec_cfg["n_fft"]
        self.hop_length = spec_cfg["hop_length"]
        self.win_length = spec_cfg["win_length"]
        self.n_mels = spec_cfg["n_mels"]
        self.f_min = spec_cfg["f_min"]
        self.f_max = spec_cfg["f_max"]
        self.power = spec_cfg.get("power", 2.0)

    # convert spectrogram into power mel spectrogram
    def to_powermel_(self, mel_spec) -> np.ndarray:
        power_mel = np.expm1(mel_spec)
        power_mel = np.maximum(power_mel, 0.0)
        
        return power_mel
    
    # convert a mel spectrogram into an approximate waveform.
    def to_waveform_(self, mel_spec) -> np.ndarray:
        power_mel = self.to_powermel_(mel_spec)

        waveform = librosa.feature.inverse.mel_to_audio(
            M=power_mel,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window="hann",
            center=True,
            pad_mode="constant",
            power=self.power,
            n_iter=self.griffinlim_iters,
            fmin=self.f_min,
            fmax=self.f_max,
        )

        return waveform

    # convert waveform to audio 
    def to_audio_(self, mel_spec) -> Audio:
        waveform = self.to_waveform_(mel_spec)
        return Audio(waveform, rate=self.sample_rate)
    
    # comparison function 
    def compare_audio(self, masked_spec, target_spec, pred_spect) -> Dict:
        masked_wave = self.to_waveform_(masked_spec)
        target_wave = self.to_waveform_(target_spec)

        print("Masked audio:")
        display(Audio(masked_wave, rate=self.sample_rate))

        print("Target audio:")
        display(Audio(target_wave, rate=self.sample_rate))