import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pywt
import matplotlib.ticker as ticker

class MelScalogram:
  def __init__(self, signal, sr, wavelet='cmor1-1', n_scales=8, lower_hz=20, upper_hz=8000):
    self.signal = signal
    self.sr = sr
    self.dt = 1/sr
    self.time = np.linspace(0, len(signal) / sr, num=len(signal))
    self.n_scales = n_scales
    self.lower_hz = lower_hz
    self.upper_hz = upper_hz
    self.freqs = self.generate_hz_points_in_mel_scale(n_scales, lower_hz, upper_hz)
    self.wavelet = wavelet
    self.scales = pywt.frequency2scale(self.wavelet, self.freqs)/self.dt
    self.freqs_mel = librosa.hz_to_mel(self.freqs)
    self.mean_subsampled_data = []
    self.mean_padding = []

  def plt_time_domain(self, figsize=(10 , 3), fname='time_domain.pdf', title='Sinal de Áudio no Domínio do Tempo'):
    '''
      Esta função faz uma plotagem do sinal ao longo do tempo.
      Parâmetros:
        figsize -> tamanho da figura: Width, height in inches.
        fname -> path para salvar a imagem.
        title -> Título para a imagem.
    '''
    plt.figure(figsize=figsize)
    plt.plot(self.time, self.signal)
    plt.title(title)
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()
    plt.savefig(fname, format='pdf')

  def generate_hz_points_in_mel_scale(self, x_points, lower_hz=0, upper_hz=8000):
      '''
      Esta função Gera x pontos de frequência em Hz distribuídos uniformemente na
      escala Mel entre os limites especificados.
      Parâmetros:
        x_points -> quantidade de pontos.
        lower_hz -> frequência mínima (Hz).
        upper_hz -> frequência máxima (Hz)
      '''
      # Converte os limites para a escala Mel
      lower_mel = librosa.hz_to_mel([lower_hz])[0]
      upper_mel = librosa.hz_to_mel([upper_hz])[0]
      # Gera x pontos na escala Mel
      mel_points = np.linspace(lower_mel, upper_mel, x_points)
      # Converte os pontos de volta para Hz
      return librosa.mel_to_hz(mel_points)


  def mean_subsample(self, data, x_points, axis=1):
      """
      Aplica subamostragem por valor médio aos dados ao longo de um eixo especificado, adicionando padding quando necessário.
      Parâmetros:
      data -> Matriz original ou vetor a ser subamostrado.
      x_points -> Número de pontos desejados após a subamostragem.
      axis -> Eixo ao longo do qual aplicar a subamostragem. Padrão é 1 (colunas).
      """
      if axis not in [0, 1]:
          raise ValueError("O eixo especificado não é válido. Use 0 para linhas ou 1 para colunas.")

      # Número de amostras no eixo especificado
      n_samples = data.shape[axis]
      # Calculando o fator de subamostragem com base no número desejado de pontos
      subsample_factor = np.ceil(n_samples / x_points).astype(int)
      # Calculando o número de pontos necessários para ter uma divisão exata
      n_needed = subsample_factor * x_points
      pad_width = n_needed - n_samples  # Calcula o padding necessário

      # Aplicando padding
      if axis == 1:  # Subamostrar ao longo das colunas
          data_padded = np.pad(data, ((0, 0), (0, pad_width)), mode='constant', constant_values=np.nan)
          new_shape = (data.shape[0], x_points, subsample_factor)
      else:  # Subamostrar ao longo das linhas
          data_padded = np.pad(data, ((0, pad_width), (0, 0)), mode='constant', constant_values=np.nan)
          new_shape = (x_points, subsample_factor, data.shape[1])

      # Redimensionando e calculando a média, ignorando NaNs
      subsampled_data = np.nanmean(data_padded.reshape(new_shape), axis=axis+1)

      return subsampled_data

  def max_subsample(self, data, x_points, axis=1):
      """
      Aplica subamostragem por valor máximo aos dados ao longo de um eixo especificado, adicionando padding quando necessário.
      Parâmetros:
      data -> Matriz original ou vetor a ser subamostrado.
      x_points -> Número de pontos desejados após a subamostragem.
      axis -> Eixo ao longo do qual aplicar a subamostragem. Padrão é 1 (colunas).
      """
      if axis not in [0, 1]:
          raise ValueError("O eixo especificado não é válido. Use 0 para linhas ou 1 para colunas.")

      # Número de amostras no eixo especificado
      n_samples = data.shape[axis]
      # Calculando o fator de subamostragem com base no número desejado de pontos
      subsample_factor = np.ceil(n_samples / x_points).astype(int)
      # Calculando o número de pontos necessários para ter uma divisão exata
      n_needed = subsample_factor * x_points
      pad_width = n_needed - n_samples  # Calcula o padding necessário

      # Aplicando padding
      if axis == 1:  # Subamostrar ao longo das colunas
          data_padded = np.pad(data, ((0, 0), (0, pad_width)), mode='constant', constant_values=np.nan)
          new_shape = (data.shape[0], x_points, subsample_factor)
      else:  # Subamostrar ao longo das linhas
          data_padded = np.pad(data, ((0, pad_width), (0, 0)), mode='constant', constant_values=np.nan)
          new_shape = (x_points, subsample_factor, data.shape[1])

      # Redimensionando e calculando a média, ignorando NaNs
      subsampled_data = np.nanmax(data_padded.reshape(new_shape), axis=axis+1)

      return subsampled_data

  def min_subsample(self, data, x_points, axis=1):
      """
      Aplica subamostragem por valor mínimo aos dados ao longo de um eixo especificado, adicionando padding quando necessário.
      Parâmetros:
      data -> Matriz original ou vetor a ser subamostrado.
      x_points -> Número de pontos desejados após a subamostragem.
      axis -> Eixo ao longo do qual aplicar a subamostragem. Padrão é 1 (colunas).
      """
      if axis not in [0, 1]:
          raise ValueError("O eixo especificado não é válido. Use 0 para linhas ou 1 para colunas.")

      # Número de amostras no eixo especificado
      n_samples = data.shape[axis]
      # Calculando o fator de subamostragem com base no número desejado de pontos
      subsample_factor = np.ceil(n_samples / x_points).astype(int)
      # Calculando o número de pontos necessários para ter uma divisão exata
      n_needed = subsample_factor * x_points
      pad_width = n_needed - n_samples  # Calcula o padding necessário

      # Aplicando padding
      if axis == 1:  # Subamostrar ao longo das colunas
          data_padded = np.pad(data, ((0, 0), (0, pad_width)), mode='constant', constant_values=np.nan)
          new_shape = (data.shape[0], x_points, subsample_factor)
      else:  # Subamostrar ao longo das linhas
          data_padded = np.pad(data, ((0, pad_width), (0, 0)), mode='constant', constant_values=np.nan)
          new_shape = (x_points, subsample_factor, data.shape[1])

      # Redimensionando e calculando a média, ignorando NaNs
      subsampled_data = np.nanmin(data_padded.reshape(new_shape), axis=axis+1)

      return subsampled_data

  def uniform_subsample(self, data, x_points, axis=1):
      """
      Aplica subamostragem uniforme aos dados ao longo de um eixo especificado, utilizando o número desejado de pontos.

      data -> Matriz original ou vetor a ser subamostrado.
      x_points -> Número desejado de pontos após a subamostragem.
      axis -> Eixo ao longo do qual aplicar a subamostragem. Padrão é 1.
                  Use 0 para subamostrar ao longo das linhas, 1 para colunas (padrão).
      :return: Dados subamostrados.
      """
      if axis not in [0, 1]:
          raise ValueError("O eixo especificado não é válido. Use 0 para linhas ou 1 para colunas.")

      # Número de amostras no eixo especificado
      n_samples = data.shape[axis]

      # Calculando o fator de subamostragem com base no número desejado de pontos
      subsample_factor = max(1, n_samples // x_points)  # Assegura que subsample_factor seja pelo menos 1

      # Aplicando a subamostragem
      if axis == 1:  # Subamostrar ao longo das colunas
          return data[:, ::subsample_factor]
      else:  # Subamostrar ao longo das linhas
          return data[::subsample_factor, :]


  def plt_mean_mel_scalogram(self, x_points=512, figsize=(5, 4), fname='mean_mel_scalogram.pdf', title="Mean Subsample (Mel-Scalogram)", n_yticks = 10):
    """
      Gera um escalograma para visualização considerando uma subamostragem por valor médio.

      x_points -> quantidade de pontos que a matriz terá após a subamostragem no eixo horizontal.
      figsize -> tamanho da figura para visualização.
      fname -> path para salvar a imagem.
      title -> Título para a imagem.
      n_yticks -> quantidade de ticks no eixo Y.
    """
    cwtmatr, freqs = pywt.cwt(self.signal, self.scales, self.wavelet, sampling_period=self.dt)
    cwtmatr = np.abs(cwtmatr)
    coef_power = cwtmatr**2
    coef_power_dB = librosa.power_to_db(coef_power, ref=np.max)
    reduced_coef_power_dB = self.mean_subsample(coef_power_dB, x_points, axis=1)
    self.mean_subsampled_data = reduced_coef_power_dB
    reduced_time = np.linspace(0, len(self.signal) / self.sr, reduced_coef_power_dB.shape[1])
    fig, axs = plt.subplots(figsize=figsize)
    pcm = axs.pcolormesh(reduced_time, self.freqs_mel, reduced_coef_power_dB)

    # Definindo explicitamente os yticks
    mel_bins = np.linspace(librosa.hz_to_mel(self.lower_hz), librosa.hz_to_mel(self.upper_hz), n_yticks)
    freq_bins_hz = librosa.mel_to_hz(mel_bins)

    # Configuração do eixo y para escala Mel
    axs.set_yscale("symlog", linthresh=librosa.hz_to_mel(1000), base=2)
    axs.set_yticks(librosa.hz_to_mel(freq_bins_hz))  # Define os ticks do eixo y nas frequências convertidas para Mel
    axs.set_yticklabels(['{:.0f}'.format(hz) for hz in freq_bins_hz])  # Rotula os ticks com os valores originais em Hertz

    axs.yaxis.set_minor_locator(plt.NullLocator()) #limpando yticks menores

    # Você pode querer ajustar os ticks do eixo y manualmente para refletir a escala Hz
    #axs.set_yticks(freqs_mel)  # Define os ticks do eixo y nas frequências convertidas para Mel
    #axs.set_yticklabels(np.round(freqs).astype(int))  # Mas rotula os ticks com os valores originais em Hertz

    axs.set_xlabel("Time (s)")
    axs.set_ylabel("Frequency (Hz)")
    axs.set_title(title)
    fig.colorbar(pcm, ax=axs, format='%+2.0f dB')
    plt.savefig(fname, format='pdf')

  def plt_uniform_mel_scalogram(self, x_points=512, figsize=(5, 4), fname='uniform_mel_scalogram.pdf', title="Uniform Subsample (Mel-Scalogram)", n_yticks=10):
    """
      Gera um escalograma para visualização considerando uma subamostragem uniforme.

      x_points -> quantidade de pontos que a matriz terá após a subamostragem no eixo horizontal.
      figsize -> tamanho da figura para visualização.
      fname -> path para salvar a imagem.
      title -> Título para a imagem.
      n_yticks -> quantidade de ticks no eixo Y.
    """
    cwtmatr, freqs = pywt.cwt(self.signal, self.scales, self.wavelet, sampling_period=self.dt)
    cwtmatr = np.abs(cwtmatr)
    coef_power = cwtmatr**2
    coef_power_dB = librosa.power_to_db(coef_power, ref=np.max)
    reduced_coef_power_dB = self.uniform_subsample(coef_power_dB, x_points, axis=1)
    reduced_time = np.linspace(0, len(self.signal) / self.sr, reduced_coef_power_dB.shape[1])
    fig, axs = plt.subplots(figsize=figsize)
    pcm = axs.pcolormesh(reduced_time, self.freqs_mel, reduced_coef_power_dB)

    # Definindo explicitamente os yticks
    mel_bins = np.linspace(librosa.hz_to_mel(self.lower_hz), librosa.hz_to_mel(self.upper_hz), n_yticks)
    freq_bins_hz = librosa.mel_to_hz(mel_bins)

    # Configuração do eixo y para escala Mel
    axs.set_yscale("symlog", linthresh=librosa.hz_to_mel(1000), base=2)
    axs.set_yticks(librosa.hz_to_mel(freq_bins_hz))  # Define os ticks do eixo y nas frequências convertidas para Mel
    axs.set_yticklabels(['{:.0f}'.format(hz) for hz in freq_bins_hz])  # Rotula os ticks com os valores originais em Hertz

    axs.yaxis.set_minor_locator(plt.NullLocator()) #limpando yticks menores

    # Você pode querer ajustar os ticks do eixo y manualmente para refletir a escala Hz
    #axs.set_yticks(freqs_mel)  # Define os ticks do eixo y nas frequências convertidas para Mel
    #axs.set_yticklabels(np.round(freqs).astype(int))  # Mas rotula os ticks com os valores originais em Hertz

    axs.set_xlabel("Time (s)")
    axs.set_ylabel("Frequency (Hz)")
    axs.set_title(title)
    fig.colorbar(pcm, ax=axs, format='%+2.0f dB')
    plt.savefig(fname, format='pdf')

  def plt_max_mel_scalogram(self, x_points=512, figsize=(5, 4), fname='max_mel_scalogram.pdf', title="Max Value Subsample (Mel-Scalogram)", n_yticks=10):
    """
      Gera um escalograma para visualização considerando uma subamostragem por valor máximo.

      x_points -> quantidade de pontos que a matriz terá após a subamostragem no eixo horizontal.
      figsize -> tamanho da figura para visualização.
      fname -> path para salvar a imagem.
      title -> Título para a imagem.
      n_yticks -> quantidade de ticks no eixo Y.
    """
    cwtmatr, freqs = pywt.cwt(self.signal, self.scales, self.wavelet, sampling_period=self.dt)
    cwtmatr = np.abs(cwtmatr)
    coef_power = cwtmatr**2
    coef_power_dB = librosa.power_to_db(coef_power, ref=np.max)
    reduced_coef_power_dB = self.max_subsample(coef_power_dB, x_points, axis=1)
    reduced_time = np.linspace(0, len(self.signal) / self.sr, reduced_coef_power_dB.shape[1])
    fig, axs = plt.subplots(figsize=figsize)
    pcm = axs.pcolormesh(reduced_time, self.freqs_mel, reduced_coef_power_dB)

    # Definindo explicitamente os yticks
    mel_bins = np.linspace(librosa.hz_to_mel(self.lower_hz), librosa.hz_to_mel(self.upper_hz), n_yticks)
    freq_bins_hz = librosa.mel_to_hz(mel_bins)

    # Configuração do eixo y para escala Mel
    axs.set_yscale("symlog", linthresh=librosa.hz_to_mel(1000), base=2)
    axs.set_yticks(librosa.hz_to_mel(freq_bins_hz))  # Define os ticks do eixo y nas frequências convertidas para Mel
    axs.set_yticklabels(['{:.0f}'.format(hz) for hz in freq_bins_hz])  # Rotula os ticks com os valores originais em Hertz

    axs.yaxis.set_minor_locator(plt.NullLocator()) #limpando yticks menores

    # Você pode querer ajustar os ticks do eixo y manualmente para refletir a escala Hz
    #axs.set_yticks(freqs_mel)  # Define os ticks do eixo y nas frequências convertidas para Mel
    #axs.set_yticklabels(np.round(freqs).astype(int))  # Mas rotula os ticks com os valores originais em Hertz

    axs.set_xlabel("Time (s)")
    axs.set_ylabel("Frequency (Hz)")
    axs.set_title(title)
    fig.colorbar(pcm, ax=axs, format='%+2.0f dB')
    plt.savefig(fname, format='pdf')

  def plt_min_mel_scalogram(self, x_points=512, figsize=(5, 4), fname='min_mel_scalogram.pdf', title="Min Value Subsample (Mel-Scalogram)", n_yticks=10):
    """
      Gera um escalograma para visualização considerando uma subamostragem por valor mínimo.

      x_points -> quantidade de pontos que a matriz terá após a subamostragem no eixo horizontal.
      figsize -> tamanho da figura para visualização.
      fname -> path para salvar a imagem.
      title -> Título para a imagem.
      n_yticks -> quantidade de ticks no eixo Y.
    """
    cwtmatr, freqs = pywt.cwt(self.signal, self.scales, self.wavelet, sampling_period=self.dt)
    cwtmatr = np.abs(cwtmatr)
    coef_power = cwtmatr**2
    coef_power_dB = librosa.power_to_db(coef_power, ref=np.max)
    reduced_coef_power_dB = self.min_subsample(coef_power_dB, x_points, axis=1)
    reduced_time = np.linspace(0, len(self.signal) / self.sr, reduced_coef_power_dB.shape[1])
    fig, axs = plt.subplots(figsize=figsize)
    pcm = axs.pcolormesh(reduced_time, self.freqs_mel, reduced_coef_power_dB)

    # Definindo explicitamente os yticks
    mel_bins = np.linspace(librosa.hz_to_mel(self.lower_hz), librosa.hz_to_mel(self.upper_hz), n_yticks)
    freq_bins_hz = librosa.mel_to_hz(mel_bins)

    # Configuração do eixo y para escala Mel
    axs.set_yscale("symlog", linthresh=librosa.hz_to_mel(1000), base=2)
    axs.set_yticks(librosa.hz_to_mel(freq_bins_hz))  # Define os ticks do eixo y nas frequências convertidas para Mel
    axs.set_yticklabels(['{:.0f}'.format(hz) for hz in freq_bins_hz])  # Rotula os ticks com os valores originais em Hertz

    axs.yaxis.set_minor_locator(plt.NullLocator()) #limpando yticks menores

    # Você pode querer ajustar os ticks do eixo y manualmente para refletir a escala Hz
    #axs.set_yticks(freqs_mel)  # Define os ticks do eixo y nas frequências convertidas para Mel
    #axs.set_yticklabels(np.round(freqs).astype(int))  # Mas rotula os ticks com os valores originais em Hertz

    axs.set_xlabel("Time (s)")
    axs.set_ylabel("Frequency (Hz)")
    axs.set_title(title)
    fig.colorbar(pcm, ax=axs, format='%+2.0f dB')
    plt.savefig(fname, format='pdf')

  def mean_for_cnn(self, x_pixels=512, y_pixels=40, cmap='gray', fname='mean_mel_scalogram_for_cnn.jpg'):
    """
      Gera um escalograma em jpg para treinamento de uma CNN, considerando uma subamostragem por valor médio.
      A possibilidade de definir a quantidade de pixels facilita a adequação à rede.

      x_pixels -> quantidade de pixels eixo X.
      y_pixels -> quantidade de pixels eixo y.
      cmap -> paleta de cores.
      fname -> caminho para salvar a imagem.
    """
    dpi=100
    inches_x=x_pixels/dpi+0.09
    inches_y=y_pixels/dpi+0.09

    cwtmatr, freqs = pywt.cwt(self.signal, self.scales, self.wavelet, sampling_period=self.dt)
    cwtmatr = np.abs(cwtmatr)
    coef_power = cwtmatr**2
    coef_power_dB = librosa.power_to_db(coef_power, ref=np.max)
    reduced_coef_power_dB = self.mean_subsample(coef_power_dB, x_pixels, axis=1)
    reduced_time = np.linspace(0, len(self.signal) / self.sr, reduced_coef_power_dB.shape[1])
    fig, ax = plt.subplots(figsize=(inches_x,inches_y), dpi=dpi,constrained_layout=True)
    fig.set_figwidth(inches_x)
    fig.set_figheight(inches_y)
    ax.axis('off')
    pcm = ax.pcolormesh(reduced_time, self.freqs_mel, reduced_coef_power_dB, cmap=cmap)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()

  def uniform_for_cnn(self, x_pixels=512, y_pixels=40, cmap='gray', fname='uniform_mel_scalogram_for_cnn.jpg'):
    """
      Gera um escalograma em jpg para treinamento de uma CNN, considerando uma subamostragem uniforme.
      A possibilidade de definir a quantidade de pixels facilita a adequação à rede.

      x_pixels -> quantidade de pixels eixo X.
      y_pixels -> quantidade de pixels eixo y.
      cmap -> paleta de cores.
      fname -> caminho para salvar a imagem.
    """
    dpi=100
    inches_x=x_pixels/dpi+0.09
    inches_y=y_pixels/dpi+0.09

    cwtmatr, freqs = pywt.cwt(self.signal, self.scales, self.wavelet, sampling_period=self.dt)
    cwtmatr = np.abs(cwtmatr)
    coef_power = cwtmatr**2
    coef_power_dB = librosa.power_to_db(coef_power, ref=np.max)
    reduced_coef_power_dB = self.uniform_subsample(coef_power_dB, x_pixels, axis=1)
    reduced_time = np.linspace(0, len(self.signal) / self.sr, reduced_coef_power_dB.shape[1])
    fig, ax = plt.subplots(figsize=(inches_x,inches_y), dpi=dpi,constrained_layout=True)
    fig.set_figwidth(inches_x)
    fig.set_figheight(inches_y)
    ax.axis('off')
    pcm = ax.pcolormesh(reduced_time, self.freqs_mel, reduced_coef_power_dB, cmap=cmap)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()

  def max_for_cnn(self, x_pixels=512, y_pixels=40, cmap='gray', fname='max_mel_scalogram_for_cnn.jpg'):
    """
      Gera um escalograma em jpg para treinamento de uma CNN, considerando uma subamostragem por valor máximo.
      A possibilidade de definir a quantidade de pixels facilita a adequação à rede.

      x_pixels -> quantidade de pixels eixo X.
      y_pixels -> quantidade de pixels eixo y.
      cmap -> paleta de cores.
      fname -> caminho para salvar a imagem.
    """
    dpi=100
    inches_x=x_pixels/dpi+0.09
    inches_y=y_pixels/dpi+0.09

    cwtmatr, freqs = pywt.cwt(self.signal, self.scales, self.wavelet, sampling_period=self.dt)
    cwtmatr = np.abs(cwtmatr)
    coef_power = cwtmatr**2
    coef_power_dB = librosa.power_to_db(coef_power, ref=np.max)
    reduced_coef_power_dB = self.max_subsample(coef_power_dB, x_pixels, axis=1)
    reduced_time = np.linspace(0, len(self.signal) / self.sr, reduced_coef_power_dB.shape[1])
    fig, ax = plt.subplots(figsize=(inches_x,inches_y), dpi=dpi,constrained_layout=True)
    fig.set_figwidth(inches_x)
    fig.set_figheight(inches_y)
    ax.axis('off')
    pcm = ax.pcolormesh(reduced_time, self.freqs_mel, reduced_coef_power_dB, cmap=cmap)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()

  def min_for_cnn(self, x_pixels=512, y_pixels=40, cmap='gray', fname='min_mel_scalogram_for_cnn.jpg'):
    """
      Gera um escalograma em jpg para treinamento de uma CNN, considerando uma subamostragem por valor mínimo.
      A possibilidade de definir a quantidade de pixels facilita a adequação à rede.

      x_pixels -> quantidade de pixels eixo X.
      y_pixels -> quantidade de pixels eixo y.
      cmap -> paleta de cores.
      fname -> caminho para salvar a imagem.
    """
    dpi=100
    inches_x=x_pixels/dpi+0.09
    inches_y=y_pixels/dpi+0.09

    cwtmatr, freqs = pywt.cwt(self.signal, self.scales, self.wavelet, sampling_period=self.dt)
    cwtmatr = np.abs(cwtmatr)
    coef_power = cwtmatr**2
    coef_power_dB = librosa.power_to_db(coef_power, ref=np.max)
    reduced_coef_power_dB = self.min_subsample(coef_power_dB, x_pixels, axis=1)
    reduced_time = np.linspace(0, len(self.signal) / self.sr, reduced_coef_power_dB.shape[1])
    fig, ax = plt.subplots(figsize=(inches_x,inches_y), dpi=dpi,constrained_layout=True)
    fig.set_figwidth(inches_x)
    fig.set_figheight(inches_y)
    ax.axis('off')
    pcm = ax.pcolormesh(reduced_time, self.freqs_mel, reduced_coef_power_dB, cmap=cmap)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close()
