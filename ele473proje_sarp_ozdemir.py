# 1 - muhru tespit et ve nerede oldugunu kaydet ardindan resimden cikar, birden fazla muhur varsa gecersiz, ilgili alandanin disindaysa gecersiz
# 2 - aday sablonunu resimden cikar
# 3 - kalan resim bembeyaz olmali, herhangi farkli bir sey varsa gecersiz
# 0 => gecersiz veya bos oy, 1 => birinci aday, 2 => ikinci aday
import cv2
import numpy as np
from PIL import Image

muhur = "muhur.png"
pusula = "pusula.png"
test_data = "test6.jpeg"
muhur_ben = "muhur_ben.jpg" #kendim kestigim kucuk muhur

templateE = cv2.imread(muhur,cv2.IMREAD_GRAYSCALE) #goruntuyu grayscale olarak yukle
templatePusula = cv2.imread(pusula,cv2.IMREAD_GRAYSCALE)


#muhru_bul fonksiyonunda kullanilacak threshold degerini bulmak icin rotasyonsuz, tek muhur basilmis pusula ile hesap:
basic_threshold_data = "test4.jpeg"
basic_verilmis_oy = cv2.imread(basic_threshold_data,cv2.IMREAD_GRAYSCALE)
basic_korelasyon = cv2.matchTemplate(basic_verilmis_oy, templateE, cv2.TM_CCOEFF_NORMED)
# "basic_korelasyon" matrisi kendim bakıp threshold degeri bulmak için çok büyük,
# bu yüzden thresholding max similarity level'ın 0.75'i ile denemeye basladim olmadı, 
# deneye deneye 0.99'a cikardim, dogru buluyor(DUZLERI, YAMUKLARI 0 SAYIYOR)
basic_similarity_data = np.max(basic_korelasyon)
threshold = 0.998 * basic_similarity_data

def e_yi_dondur(image_E, angle):
    height, width = image_E.shape[:2]
    center = (width // 2, height // 2)
    
    # Pad the image with white border to preserve its size
    max_dim = max(height, width)
    pad_x = (max_dim - width) // 2
    pad_y = (max_dim - height) // 2
    padded_img = cv2.copyMakeBorder(image_E, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=255)
    
    # Rotate the padded image
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_img = cv2.warpAffine(padded_img, rotation_matrix, (max_dim, max_dim), borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    
    return rotated_img

def e_yi_kirp(image, crop_width, crop_height, side='center'):
    height, width = image.shape[:2]
    
    if side == 'left':
        crop_left = 0
        crop_right = crop_width
    elif side == 'right':
        crop_left = width - crop_width
        crop_right = width
    else:  # Default to center cropping
        crop_left = (width - crop_width) // 2
        crop_right = crop_left + crop_width
    
    crop_top = (height - crop_height) // 2
    crop_bottom = crop_top + crop_height
    
    cropped_image = image[crop_top:crop_bottom, crop_left:crop_right]
    
    return cropped_image


def muhru_bul(oy_pusulasi):#gecersizse print(gecersiz) return 0, gecerliyse return 1 veya 2 (adaylar) 
    verilmis_oy = cv2.imread(oy_pusulasi,cv2.IMREAD_GRAYSCALE)
    #correlation of images islemini uygula
    #korelasyon = cv2.matchTemplate(verilmis_oy, templateE, cv2.TM_CCOEFF_NORMED)
    #cv2.TM_CCOEFF_NORMED => normalize edilmis cross-correlation katsayilarini hesaplar
    e_bulundu = []
    
    #yamuk icin
    for aci in range(360):
        '''rotatedE = e_yi_dondur(templateE,aci)
        korelasyon = cv2.matchTemplate(verilmis_oy, rotatedE, cv2.TM_CCOEFF_NORMED)
        locations_E = np.where(korelasyon >= threshold)
        e_bulundu.extend(list(zip(*locations_E[::-1])))'''
        rotatedE = e_yi_dondur(templateE,aci)
        if(aci == 0):
            croppedE = rotatedE
        elif(aci == 90):
            temp = e_yi_kirp(rotatedE,101,90,side = 'left')
            croppedE = e_yi_kirp(temp,101,85,side = 'down')
        elif(aci == 180):
            croppedE= e_yi_kirp(rotatedE,70,100,side = 'left')
        elif((0 < aci) & (aci < 90)):
            croppedE = e_yi_kirp(rotatedE,112,115, side ='down')
        elif((90 < aci) & (aci < 180)):
            croppedE = rotatedE
        elif((180 < aci) & (aci < 270)):
            croppedE = e_yi_kirp(rotatedE,99,124, side ='left')
        elif((270 < aci) & (aci < 360)):
            croppedE = e_yi_kirp(rotatedE,111,124, side ='up')
        
        korelasyon = cv2.matchTemplate(verilmis_oy,croppedE,cv2.TM_CCOEFF_NORMED)
        locationsE = np.where(korelasyon >= threshold)
        e_bulundu.extend(list(zip(*locationsE[::-1])))

    num_matched_templates = len(e_bulundu)
    #print("Bulunan muhur sayisi:",num_matched_templates)

    if(num_matched_templates != 1):# bos oy atilmis veya birden fazla muhur basilmissa
        return 0,[],[],[]
    else:#gecerli oy oldugunu buldugumuza gore(muhur bakımından) oyun yerine bakalım
        #duz E icin her durumu top left ve bottom right ile cozebiliyoruz
        
        template_width = templateE.shape[1]
        template_height = templateE.shape[0]
        top_left = min(e_bulundu, key=lambda p: p[0] + p[1])
        bottom_right = (top_left[0] + template_width, top_left[1] + template_height)

        top_right = (top_left[0] + template_width, top_left[1])
        bottom_left = (top_left[0], top_left[1] + template_height)
        return top_left, top_right, bottom_left, bottom_right

def adayi_bul(top_left, top_right, bottom_left, bottom_right):
    '''pusula.png'ye gore bulunmus degerler(el ile)
        1. adayin kutucugu (133,94) sol ust kose,(338,505) sag alt kose 
        2. adayin kutucugu (444,94) sol ust kose,(649,505) sag alt kose 
        1. adaydaki 1 yazisi (156,116) sol ust kose, (316,309) sag alt kose
        2. adaydaki 2 yazisi (467,309) sol ust kose, (627,309) sag alt kose
        dolayisiyla 1. adaydaki available muhur alani 133 <= x <= 338, 309 <= y <= 505
        2. adaydaki available muhur alani 444 <= x <= 649, 309 <= y <= 505'''
    sol_x = top_left[0]
    sag_x = top_right[0]
    ust_y = top_left[1]
    alt_y = bottom_right[1]
    if((sol_x >= 133 and sag_x <= 338) and (ust_y >= 309 and alt_y <= 505)):
        return 1
    elif((sol_x >= 444 and sag_x <= 649) and (ust_y >= 309 and alt_y <= 505)):
        return 2
    else:
        return 0

def muhuru_sil(top_left, top_right, bottom_left, bottom_right,oy_pusulasi):#muhurun bulundugu rectangle'yi sil
    muhurlu_pusula = cv2.imread(oy_pusulasi,cv2.IMREAD_GRAYSCALE)
    sol_x = top_left[0]
    sag_x = top_right[0]
    ust_y = top_left[1]
    alt_y = bottom_right[1]
    muhursuz_pusula = muhurlu_pusula.copy()
    muhursuz_pusula[top_left[1]-1:bottom_right[1]+1, top_left[0]-1:bottom_right[0]+1] = 255 #1'ler yamuk E'lerde silmeyi garantiye almak için
    cv2.imshow('Muhursuz Pusula',muhursuz_pusula)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return muhursuz_pusula

def karsilastirma(muhursuz_pusula, duz_pusula):
    #muhursuz = cv2.imread(muhursuz_pusula, cv2.IMREAD_GRAYSCALE)
    pusula = cv2.imread(duz_pusula, cv2.IMREAD_GRAYSCALE)

    # Check if the images have the same dimensions
    if muhursuz_pusula.shape != pusula.shape:
        print('esit olmali')
        return False

    difference = cv2.absdiff(muhursuz_pusula, pusula)    
    _, thresholded = cv2.threshold(difference,230, 255, cv2.THRESH_BINARY)    
    num_non_zero = np.count_nonzero(thresholded)

    return num_non_zero == 0

koseler = muhru_bul(test_data)
if(koseler[0] == 0 ):
    print('Gecersiz veya Bos Oy, muhur')
else:    
    print("Sol Ust:", koseler[0])
    print("Sag Ust:", koseler[1])
    print("Sol Alt:", koseler[2])
    print("Sag Alt:", koseler[3])
    aday = adayi_bul(koseler[0],koseler[1],koseler[2],koseler[3])
    muhursuz_pusula = muhuru_sil(koseler[0],koseler[1],koseler[2],koseler[3],test_data)
    a = karsilastirma(muhursuz_pusula,pusula)
    if a:
        if aday == 1:
            print('Aday 1')
        elif aday == 2:
            print('Aday 2')
    else:
        print('Gecersiz veya Bos Oy')
