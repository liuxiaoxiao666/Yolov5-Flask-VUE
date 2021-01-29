import cv2

def predict(dataset, model, ext):
    global img_y
    x = dataset[0].replace('\\', '/')
    file_name = dataset[1]
    print(x)
    print(file_name)
    x = cv2.imread(x)
    img_y, image_info = model.detect(x)
    cv2.imwrite('./tmp/draw/{}.{}'.format(file_name, ext), img_y)
    return image_info