from ctpnport import *
from crnnport import *

# ctpn
text_detector = ctpnSource()
# crnn
model, converter = crnnSource()

timer = Timer()
print("\ninput exit break\n")
while 1:
    im_name = raw_input("\nplease input file name:")
    if im_name == "exit":
        break
    im_path = './img/' + im_name
    im = cv2.imread(im_path)
    if im is None:
        continue
    timer.tic()
    # 执行ctpn检测，得到图片上文字的boxes labels
    # text_detecotor是检测器（含文本线构造器）
    # text_recs是text boxes:(x1, y1, x2, y2, x3, y3, x4, y4)
    img, text_recs = getCharBlock(text_detector, im)

    # 执行crnn识别，得到图片上的文字labels
    crnnRec(model, converter, img, text_recs)
