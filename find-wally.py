import cv2

#Lee las imágenes
img = cv2.imread("images\where-is-wally.jpg")
wally = cv2.imread("images\Wally.jpg")

# Escala las imagenes de fondo para que pueda verse en pantalla
scale_percent = 15
width_img = int(img.shape[1] * scale_percent / 100)
height_img = int(img.shape[0] * scale_percent / 100)
dim_img = (width_img, height_img)
img_resized = cv2.resize(img, dim_img)

width_wally = int(wally.shape[1] * scale_percent / 100)
height_wally = int(wally.shape[0] * scale_percent / 100)
dim_wally = (width_wally, height_wally)
wally_resized = cv2.resize(wally, dim_wally)

#Convierte las imágenes de RGB a escala de grises para que opencv pueda trabajar con ellas
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
wally_gray = cv2.cvtColor(wally_resized, cv2.COLOR_BGR2GRAY)

# Compara la imagen de wally con la imagen de fondo, buscando puntos en común. 
# Más información: https://docs.opencv.org/3.4/de/da9/tutorial_template_matching.html
#                  https://docs.opencv.org/4.5.2/d4/dc6/tutorial_py_template_matching.html
matrix = cv2.matchTemplate(img_gray, wally_gray, cv2.TM_SQDIFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matrix)  # Mínimo y máximo de la matriz y sus posiciones
print(min_val, max_val, min_loc, max_loc)

# Dibujar rectángulo
x1, y1 = min_loc
x2, y2 = min_loc[0] + wally_resized.shape[1], min_loc[1] + wally_resized.shape[0]
cv2.rectangle(img_resized, (x1,y1), (x2,y2), (0,0,0), 3)

# Muestra la imagen
cv2.imshow("Where's Wally?", img_resized)
cv2.waitKey(0)