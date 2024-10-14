# pip install pytesseract
# sudo apt install tesseract-ocr-tur
# pip install -q git+https://github.com/THU-MIG/yolov10.git
# pip install -q supervision

import cv2
import supervision as sv
from ultralytics import YOLOv10
import os
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



def main():
    model = YOLOv10('yolov10x_best.pt')
    folder_path = 'D:\YUSUF\Staj\BelgeOrnekleri\images'#Resimlerin bulunduğu klasör
    results_folder_path = 'D:\YUSUF\Staj\BelgeOrnekleri\results'#OCR sonuçlarının kaydedileceği klasör

    process_files_in_folder(folder_path, results_folder_path, model)


    


def process_files_in_folder(folder_path, results_folder_path, model):
    # Klasördeki tüm dosyaları al
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        # Burada dosya ile ilgili işlemleri yapabilirsiniz

        layout_process(image_path, results_folder_path, model)





def layout_process(image_path, results_folder_path, model):
    image = cv2.imread(image_path)

    results = model(source=image_path, conf=0.2, iou=0.8)[0]

    detections = sv.Detections.from_ultralytics(results)
    sorted_detections = sorted(detections, key=lambda detection: detection[0][1])
    new_detections = filter_detections(sorted_detections)

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    #sv.plot_image(annotated_image)

    OCR_Result(new_detections,image_path,results_folder_path)

    





def filter_detections(sorted_detections):
    import numpy as np

    new_detections = []

    i = 0
    while i < len(sorted_detections)-1:
        #print(sorted_detections[i][0])
        #print(i)
        if(np.array_equal(sorted_detections[i][0],sorted_detections[i+1][0])):
            #print(sorted_detections[i][0])
            #print(sorted_detections[i][5])
            #print(sorted_detections[i+1][5])
            if(sorted_detections[i][5] == {'class_name': 'Text'}):
                new_detections.append(sorted_detections[i])
            elif(sorted_detections[i+1][5] == {'class_name': 'Text'}):
                new_detections.append(sorted_detections[i+1])
            elif(sorted_detections[i][5] == {'class_name': 'Picture'}):
                new_detections.append(sorted_detections[i+1])
            else:
                new_detections.append(sorted_detections[i])
            i = i + 1
        else:
            new_detections.append(sorted_detections[i])
        i = i + 1

        if(i == len(sorted_detections)-1):
            new_detections.append(sorted_detections[i])

    

    for detection in new_detections:
        for detection2 in new_detections:
            x0, y0, x1, y1 = detection[0]
            x02, y02, x12, y12 = detection2[0]
            if(x0 > x02 and y0 > y02 and x1 < x12 and y1 < y12):
                detection[0][1] = -1.00
                #print(detection)
                #print(detection2)


    return new_detections




def OCR_Result(new_detections, image_path, results_folder_path):

    from PIL import Image
    import pytesseract
    import math

    img_copy = cv2.imread(image_path)
    text=""

    for detection in new_detections:
        # detection objesinden label bilgisini al ve yazdır
        x0, y0, x1, y1 = detection[0]
        if not(y0 > 0):
            continue
        cropped_image = img_copy[math.floor(y0):math.floor(y1), math.ceil(x0):math.ceil(x1)]

        img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = Image.fromarray(img)

        #sv.plot_image(cropped_image)

        if(detection[5] == {'class_name': 'Table'}):
            
            ornek = 0
            text += table_process(cropped_image)

        else:
            text += pytesseract.image_to_string(cropped_image, lang="tur")
            
    print(text)


        #cropped_image.close()
        #os.remove(cropped_image_path)

def table_process(cropped_image):

    from img2table.document import Image
    from img2table.ocr import TesseractOCR
    import cv2
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    ocr = TesseractOCR(n_threads=1, lang="tur")

    img_tables = cropped_image.extract_tables(ocr=ocr,
                                implicit_rows=False,
                                implicit_columns=False,
                                borderless_tables=False,
                                min_confidence=50)
    
    string=""
    for table in img_tables:
        for id_row, row in enumerate(table.content.values()):
            #print(len(row))
            for id_col, cell in enumerate(row):
                x0 = cell.bbox.x1
                y0 = cell.bbox.y1
                x1 = cell.bbox.x2
                y1 = cell.bbox.y2

                cropped_table = cropped_image[y0:y1, x0:x1]

                img2 = cv2.cvtColor(cropped_table, cv2.COLOR_BGR2RGB)
                cell_image = Image.fromarray(img2)
                text = pytesseract.image_to_string(cell_image, lang="tur")
                formatted_string = " ".join(text.split())
                if formatted_string == "":
                    formatted_string = "None"
                string += formatted_string + ","
            string += "\n"

    return string





if __name__ == "__main__":
    main()