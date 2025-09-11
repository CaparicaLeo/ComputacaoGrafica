# -----------------------------------------------------------------------------
# Processamento Digital de Imagens - Lista de Exercícios 2
# Implementação de Filtros no Domínio Espacial
# -----------------------------------------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Passo 0: Funções Auxiliares e Carregamento de Imagens ---

def mostrar_imagens(original, processada, titulo_original='Original', titulo_processada='Processada'):
    """
    Função para exibir a imagem original e a processada lado a lado.
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title(titulo_original)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(processada, cmap='gray')
    plt.title(titulo_processada)
    plt.axis('off')
    
    # Para salvar as imagens para o PDF, descomente a linha abaixo
    # plt.savefig(f"{titulo_processada.replace(' ', '_')}.png")
    
    plt.show()

# Tenta carregar as imagens. Altere 'img_aluno.png' se o nome for diferente.
try:
    img_lena = cv2.imread('Lista2/assets/lena.png')
    img_aluno = cv2.imread('Lista2/assets/img_aluno.jpg') # <-- ATENÇÃO: Altere aqui se necessário
    if img_lena is None or img_aluno is None:
        raise FileNotFoundError
except FileNotFoundError:
    print("ERRO: Verifique se 'lena.png' e 'img_aluno.png' estão na mesma pasta do script.")
    exit()

# Converte para níveis de cinza, conforme solicitado na atividade
lena_cinza = cv2.cvtColor(img_lena, cv2.COLOR_BGR2GRAY)
aluno_cinza = cv2.cvtColor(img_aluno, cv2.COLOR_BGR2GRAY)


# --- Questão 1: Filtro de Suavização pela Média ---

def filtro_media(imagem, tamanho_janela):
    """Aplica um filtro de suavização pela média."""
    if tamanho_janela % 2 == 0:
        raise ValueError("O tamanho da janela deve ser um número ímpar.")
    img_filtrada = np.zeros_like(imagem, dtype=np.float32)
    altura, largura = imagem.shape
    pad = tamanho_janela // 2
    img_padded = cv2.copyMakeBorder(imagem, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    
    for y in range(altura):
        for x in range(largura):
            vizinhanca = img_padded[y : y + tamanho_janela, x : x + tamanho_janela]
            media = np.mean(vizinhanca)
            img_filtrada[y, x] = media
            
    return img_filtrada.astype(np.uint8)


# --- Questão 2: Filtro de Suavização pela Média dos k Vizinhos Mais Próximos ---

def filtro_k_vizinhos_proximos(imagem, tamanho_janela, k):
    """Aplica um filtro pela média dos k vizinhos mais próximos em intensidade."""
    if tamanho_janela % 2 == 0:
        raise ValueError("O tamanho da janela deve ser um número ímpar.")
    if k > tamanho_janela**2:
         raise ValueError("k não pode ser maior que o número de elementos na janela.")

    img_filtrada = np.zeros_like(imagem, dtype=np.float32)
    altura, largura = imagem.shape
    pad = tamanho_janela // 2
    img_padded = cv2.copyMakeBorder(imagem, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    
    for y in range(altura):
        for x in range(largura):
            vizinhanca = img_padded[y : y + tamanho_janela, x : x + tamanho_janela]
            pixel_central = vizinhanca[pad, pad]
            vizinhos_flat = vizinhanca.flatten()
            diferencas = np.abs(vizinhos_flat - pixel_central)
            indices_proximos = np.argsort(diferencas)[:k]
            valores_proximos = vizinhos_flat[indices_proximos]
            media_k = np.mean(valores_proximos)
            img_filtrada[y, x] = media_k
            
    return img_filtrada.astype(np.uint8)


# --- Questão 3: Filtro de Suavização pela Mediana ---

def filtro_mediana(imagem, tamanho_janela):
    """Aplica um filtro de suavização pela mediana."""
    if tamanho_janela % 2 == 0:
        raise ValueError("O tamanho da janela deve ser um número ímpar.")
        
    img_filtrada = np.zeros_like(imagem, dtype=np.uint8)
    altura, largura = imagem.shape
    pad = tamanho_janela // 2
    img_padded = cv2.copyMakeBorder(imagem, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    
    for y in range(altura):
        for x in range(largura):
            vizinhanca = img_padded[y : y + tamanho_janela, x : x + tamanho_janela]
            mediana = np.median(vizinhanca)
            img_filtrada[y, x] = mediana
            
    return img_filtrada


# --- Questão 4: Operador Laplaciano ---

def filtro_laplaciano(imagem):
    """Aplica o operador Laplaciano para detecção de bordas."""
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    img_laplace = cv2.filter2D(imagem, -1, kernel)
    img_laplace_abs = cv2.convertScaleAbs(img_laplace)
    return img_laplace_abs


# --- Questão 5: Detector de Bordas de Roberts ---

def detector_roberts(imagem):
    """Aplica o detector de bordas de Roberts."""
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    grad_x = cv2.filter2D(imagem, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(imagem, cv2.CV_64F, kernel_y)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return magnitude_norm


# --- Questão 6: Detector de Bordas de Prewitt ---

def detector_prewitt(imagem):
    """Aplica o detector de bordas de Prewitt."""
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    grad_x = cv2.filter2D(imagem, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(imagem, cv2.CV_64F, kernel_y)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return magnitude_norm


# --- Questão 7: Detector de Bordas de Sobel ---

def detector_sobel(imagem):
    """Aplica o detector de bordas de Sobel."""
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    grad_x = cv2.filter2D(imagem, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(imagem, cv2.CV_64F, kernel_y)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return magnitude_norm


# --- Bloco Principal de Execução ---
if __name__ == "__main__":
    print("Iniciando a execução da Lista de Exercícios 2...")
    print("Feche cada janela de imagem para continuar para a próxima questão.")

    # --- Execução da Questão 1 ---
    print("\nExecutando Questão 1: Filtro da Média...")
    lena_media = filtro_media(lena_cinza, 5)
    mostrar_imagens(lena_cinza, lena_media, titulo_processada='Lena - Filtro da Media 5x5')
    aluno_media = filtro_media(aluno_cinza, 5)
    mostrar_imagens(aluno_cinza, aluno_media, titulo_processada='Aluno - Filtro da Media 5x5')
    
    # --- Execução da Questão 2 ---
    print("\nExecutando Questão 2: Filtro K Vizinhos Mais Próximos...")
    lena_k_vizinhos = filtro_k_vizinhos_proximos(lena_cinza, 5, 10)
    mostrar_imagens(lena_cinza, lena_k_vizinhos, titulo_processada='Lena - Filtro K Vizinhos (k=10)')
    aluno_k_vizinhos = filtro_k_vizinhos_proximos(aluno_cinza, 5, 10)
    mostrar_imagens(aluno_cinza, aluno_k_vizinhos, titulo_processada='Aluno - Filtro K Vizinhos (k=10)')

    # --- Execução da Questão 3 ---
    print("\nExecutando Questão 3: Filtro da Mediana...")
    lena_mediana = filtro_mediana(lena_cinza, 3)
    mostrar_imagens(lena_cinza, lena_mediana, titulo_processada='Lena - Filtro da Mediana 3x3')
    aluno_mediana = filtro_mediana(aluno_cinza, 3)
    mostrar_imagens(aluno_cinza, aluno_mediana, titulo_processada='Aluno - Filtro da Mediana 3x3')

    # --- Execução da Questão 4 ---
    print("\nExecutando Questão 4: Operador Laplaciano...")
    lena_laplace = filtro_laplaciano(lena_cinza)
    mostrar_imagens(lena_cinza, lena_laplace, titulo_processada='Lena - Operador Laplaciano')
    aluno_laplace = filtro_laplaciano(aluno_cinza)
    mostrar_imagens(aluno_cinza, aluno_laplace, titulo_processada='Aluno - Operador Laplaciano')

    # --- Execução da Questão 5 ---
    print("\nExecutando Questão 5: Detector de Roberts...")
    lena_roberts = detector_roberts(lena_cinza)
    mostrar_imagens(lena_cinza, lena_roberts, titulo_processada='Lena - Detector de Roberts')
    aluno_roberts = detector_roberts(aluno_cinza)
    mostrar_imagens(aluno_cinza, aluno_roberts, titulo_processada='Aluno - Detector de Roberts')

    # --- Execução da Questão 6 ---
    print("\nExecutando Questão 6: Detector de Prewitt...")
    lena_prewitt = detector_prewitt(lena_cinza)
    mostrar_imagens(lena_cinza, lena_prewitt, titulo_processada='Lena - Detector de Prewitt')
    aluno_prewitt = detector_prewitt(aluno_cinza)
    mostrar_imagens(aluno_cinza, aluno_prewitt, titulo_processada='Aluno - Detector de Prewitt')
    
    # --- Execução da Questão 7 ---
    print("\nExecutando Questão 7: Detector de Sobel...")
    lena_sobel = detector_sobel(lena_cinza)
    mostrar_imagens(lena_cinza, lena_sobel, titulo_processada='Lena - Detector de Sobel')
    aluno_sobel = detector_sobel(aluno_cinza)
    mostrar_imagens(aluno_cinza, aluno_sobel, titulo_processada='Aluno - Detector de Sobel')

    print("\n--- Execução Concluída ---")