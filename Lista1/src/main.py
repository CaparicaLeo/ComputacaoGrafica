import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- FUNÇÕES AUXILIARES PARA EXIBIÇÃO ---

def plot_result(img_original, img_processed, title_original='Original', title_processed='Processada'):
    """Exibe a imagem original e a processada lado a lado."""
    # Converte imagens de BGR (OpenCV) para RGB (Matplotlib) se forem coloridas
    if len(img_original.shape) == 3:
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    if len(img_processed.shape) == 3:
        img_processed = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_original, cmap='gray' if len(img_original.shape) == 2 else None)
    plt.title(title_original)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_processed, cmap='gray' if len(img_processed.shape) == 2 else None)
    plt.title(title_processed)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# --- EXERCÍCIO 1: CONVERSÃO PARA NÍVEIS DE CINZA ---

def to_grayscale(image):
    """Converte uma imagem colorida para níveis de cinza."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- EXERCÍCIO 2: NEGATIVO DA IMAGEM ---

def get_negative(image):
    """Gera o negativo de uma imagem."""
    # L-1 - r, onde L=256 para imagens de 8 bits
    return 255 - image

# --- EXERCÍCIO 3: AJUSTE DE CONTRASTE (NORMALIZAÇÃO) ---

def adjust_contrast(image, new_min, new_max):
    """Normaliza o contraste da imagem para uma nova faixa de valores."""
    # cv2.normalize(src, dst, alpha, beta, norm_type)
    # alpha = new_min, beta = new_max
    return cv2.normalize(image, None, alpha=new_min, beta=new_max, norm_type=cv2.NORM_MINMAX)

# --- EXERCÍCIO 4: OPERADOR LOGARÍTMICO ---

def log_transform(image):
    """Aplica a transformação logarítmica na imagem."""
    # c = 255 / log(1 + max_pixel_value)
    c = 255 / np.log(1 + np.max(image))
    # s = c * log(1 + r)
    log_image = c * (np.log(image + 1))
    return np.array(log_image, dtype=np.uint8)

# --- EXERCÍCIO 5: OPERADOR DE POTÊNCIA (GAMMA) ---

def power_law_transform(image, c, gamma):
    """Aplica a transformação de potência (gamma) na imagem."""
    # s = c * r^gamma
    # Normaliza a imagem para [0, 1]
    image_normalized = image / 255.0
    # Aplica a transformação
    gamma_corrected = c * np.power(image_normalized, gamma)
    # Escala de volta para [0, 255]
    return np.array(gamma_corrected * 255, dtype=np.uint8)

# --- EXERCÍCIO 6: FATIAMENTO DOS PLANOS DE BITS ---

def bit_plane_slicing(gray_image):
    """Separa uma imagem em seus 8 planos de bits."""
    planes = []
    for k in range(8):
        # Extrai o k-ésimo bit e o torna o mais significativo para visualização
        plane = (gray_image >> k) & 1
        planes.append(plane * 255)
    return planes

def plot_bit_planes(planes):
    """Exibe os 8 planos de bits."""
    plt.figure(figsize=(12, 6))
    for i, plane in enumerate(planes):
        plt.subplot(2, 4, i + 1)
        plt.imshow(plane, cmap='gray')
        plt.title(f'Plano de Bit {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# --- EXERCÍCIO 7: HISTOGRAMAS ---

def calculate_histogram(image, channel=0):
    """Calcula o histograma de uma imagem (ou de um canal específico)."""
    if len(image.shape) == 2: # Grayscale
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    else: # Color
        hist = cv2.calcHist([image], [channel], None, [256], [0, 256])
    return hist

def normalize_histogram(hist):
    """Normaliza um histograma."""
    return hist / hist.sum()

def cumulative_histogram(hist):
    """Calcula o histograma acumulado."""
    return hist.cumsum()

def plot_histograms(hist_dict, title="Histogramas"):
    """Plota múltiplos histogramas."""
    plt.figure(figsize=(10, 8))
    plt.suptitle(title)
    
    plot_index = 1
    for label, (hist_data, color) in hist_dict.items():
        plt.subplot(len(hist_dict), 1, plot_index)
        plt.plot(hist_data, color=color)
        plt.title(label)
        plt.xlim([0, 256])
        plot_index += 1
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- EXERCÍCIO 8: EQUALIZAÇÃO DE HISTOGRAMA ---

def equalize_histogram(image):
    """Aplica equalização de histograma."""
    if len(image.shape) == 2: # Grayscale
        return cv2.equalizeHist(image)
    else: # Color
        # Converte para YCrCb, equaliza o canal Y (luminância) e converte de volta
        ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        return cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

# --- FUNÇÃO PRINCIPAL PARA EXECUÇÃO DOS EXERCÍCIOS ---

def main():
    # Carregar imagens
    try:
        img_lena = cv2.imread('Lista1/assets/lena.png', cv2.IMREAD_COLOR)
        img_aluno = cv2.imread('Lista1/assets/img_aluno.jpg', cv2.IMREAD_COLOR)
        img_unequalized = cv2.imread('Lista1/assets/unequalized.jpg', cv2.IMREAD_COLOR)

        # Redimensiona img_aluno se necessário (conforme Obs. 1)
        img_aluno = cv2.resize(img_aluno, (500, 500))

    except Exception as e:
        print(f"Erro ao carregar as imagens. Verifique se os arquivos estão na pasta correta.")
        print(f"Detalhe do erro: {e}")
        return

    # --- Exercício 1 ---
    print("Executando Exercício 1: Níveis de Cinza")
    gray_lena = to_grayscale(img_lena)
    gray_aluno = to_grayscale(img_aluno)
    plot_result(img_lena, gray_lena, title_processed='Lena - Níveis de Cinza')
    plot_result(img_aluno, gray_aluno, title_processed='Img Aluno - Níveis de Cinza')

    # --- Exercício 2 ---
    print("\nExecutando Exercício 2: Negativo")
    negative_lena = get_negative(img_lena)
    negative_aluno = get_negative(img_aluno)
    plot_result(img_lena, negative_lena, title_processed='Lena - Negativo')
    plot_result(img_aluno, negative_aluno, title_processed='Img Aluno - Negativo')

    # --- Exercício 3 ---
    print("\nExecutando Exercício 3: Ajuste de Contraste")
    contrast_lena = adjust_contrast(img_lena, 0, 100)
    contrast_aluno = adjust_contrast(img_aluno, 0, 100)
    plot_result(img_lena, contrast_lena, title_processed='Lena - Contraste (0-100)')
    plot_result(img_aluno, contrast_aluno, title_processed='Img Aluno - Contraste (0-100)')
    
    # --- Exercício 4 ---
    print("\nExecutando Exercício 4: Operador Logarítmico")
    log_lena = log_transform(to_grayscale(img_lena)) # Operador usualmente aplicado em grayscale
    log_aluno = log_transform(to_grayscale(img_aluno))
    plot_result(to_grayscale(img_lena), log_lena, title_processed='Lena - Logarítmico')
    plot_result(to_grayscale(img_aluno), log_aluno, title_processed='Img Aluno - Logarítmico')

    # --- Exercício 5 ---
    print("\nExecutando Exercício 5: Operador de Potência")
    power_lena = power_law_transform(to_grayscale(img_lena), c=2, gamma=2)
    power_aluno = power_law_transform(to_grayscale(img_aluno), c=2, gamma=2)
    plot_result(to_grayscale(img_lena), power_lena, title_processed='Lena - Potência (c=2, gamma=2)')
    plot_result(to_grayscale(img_aluno), power_aluno, title_processed='Img Aluno - Potência (c=2, gamma=2)')

    # --- Exercício 6 ---
    print("\nExecutando Exercício 6: Fatiamento de Planos de Bits")
    planes_lena = bit_plane_slicing(gray_lena)
    planes_aluno = bit_plane_slicing(gray_aluno)
    plot_bit_planes(planes_lena)
    plot_bit_planes(planes_aluno)

    # --- Exercício 7 ---
    print("\nExecutando Exercício 7: Histogramas")
    # (i) Histograma de unequalized.jpg em cinza
    gray_unequalized = to_grayscale(img_unequalized)
    hist_unequalized = calculate_histogram(gray_unequalized)
    plot_histograms({'Histograma Unequalized (Cinza)': (hist_unequalized, 'black')}, title="7-i")

    # (ii) Histogramas dos canais R, G, B de img_aluno
    b, g, r = cv2.split(img_aluno)
    hist_b = calculate_histogram(b)
    hist_g = calculate_histogram(g)
    hist_r = calculate_histogram(r)
    plot_histograms({
        'Canal Azul (B)': (hist_b, 'blue'),
        'Canal Verde (G)': (hist_g, 'green'),
        'Canal Vermelho (R)': (hist_r, 'red')
    }, title="7-ii: Histogramas RGB - img_aluno")

    # (iii) Histogramas A, B, C, D de img_aluno em cinza
    hist_A = calculate_histogram(gray_aluno)
    hist_B = normalize_histogram(hist_A)
    hist_C = cumulative_histogram(hist_A)
    hist_D = cumulative_histogram(hist_B)
    plot_histograms({
        'A: Histograma': (hist_A, 'black'),
        'B: Normalizado': (hist_B, 'blue'),
        'C: Acumulado': (hist_C, 'green'),
        'D: Acumulado Normalizado': (hist_D, 'red')
    }, title="7-iii: Histogramas - img_aluno (Cinza)")

    # --- Exercício 8 ---
    print("\nExecutando Exercício 8: Equalização de Histograma")
    eq_lena = equalize_histogram(img_lena)
    eq_unequalized = equalize_histogram(img_unequalized)
    eq_aluno = equalize_histogram(img_aluno)
    plot_result(img_lena, eq_lena, title_processed='Lena - Equalizada')
    plot_result(img_unequalized, eq_unequalized, title_processed='Unequalized - Equalizada')
    plot_result(img_aluno, eq_aluno, title_processed='Img Aluno - Equalizada')


if __name__ == '__main__':
    main()