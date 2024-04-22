import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import os

# Descargar recursos necesarios para NLTK
nltk.download('punkt')

def extraer_texto(pdf_file):
    # Verificar si el archivo PDF existe
    if not os.path.exists(pdf_file):
        print("El archivo PDF especificado no existe.")
        return None
    
    # Abrir el archivo PDF en modo lectura binaria
    with open(pdf_file, 'rb') as file:
        # Crear un objeto PDFReader
        pdf_reader = PyPDF2.PdfFileReader(file)
        # Inicializar una cadena vacía para almacenar el texto extraído
        text = ''
        # Iterar sobre cada página del PDF
        for page_num in range(pdf_reader.numPages):
            # Obtener el objeto Page correspondiente a la página actual
            page = pdf_reader.getPage(page_num)
            # Extraer el texto de la página actual y añadirlo a la cadena de texto
            text += page.extractText()
    return text

def procesar_texto(texto):
    # Tokenizar el texto en palabras
    palabras = word_tokenize(texto)
    # Calcular el número total de palabras
    total_palabras = len(palabras)
    # Calcular la frecuencia de cada palabra
    freq_dist = FreqDist(palabras)
    # Obtener las palabras que aparecen una sola vez
    palabras_unicas = freq_dist.hapaxes()
    # Graficar las 20 palabras más comunes
    freq_dist.plot(20, cumulative=False)
    plt.show()
    
    return total_palabras, palabras_unicas, freq_dist

def main():
    # Ruta del archivo PDF
    pdf_file = 'Conceptos.pdf'
    
    # Extraer texto del PDF
    texto = extraer_texto(pdf_file)
    
    # Si no se pudo extraer el texto, terminar el programa
    if texto is None:
        return
    
    # Procesar el texto
    total_palabras, palabras_unicas, freq_dist = procesar_texto(texto)
    
    # Mostrar resultados
    print("Total de palabras:", total_palabras)
    print("Palabras únicas:", palabras_unicas)
    print("Distribución de frecuencia:", freq_dist)

if __name__ == "__main__":
    main()
