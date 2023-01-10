import random
import time
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont
from .m_types import *

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]


class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]

class IMAGE(Structure):
    _fields_ = [("nr", c_int),
                ("nc", c_int),
                ("nch", c_int),
                ("data", POINTER(c_float))]


class METADATA(Structure):
    _fields_ = [("n", c_int),
                ("names", POINTER(c_char_p))]


class Math21Recognize(object):
    def __init__(self):
        star_lib = CDLL(star_lib_path, RTLD_GLOBAL)

        self.star_ml_function_rnn_train = star_lib.star_ml_function_rnn_train
        self.star_ml_function_rnn_train.argtypes = [c_char_p, c_char_p, c_char_p, c_int, c_int]

        # predict = star_lib.math21_ml_net_predict_input
        predict = star_lib.math21_ml_function_net_predict_input
        predict.argtypes = [c_void_p, POINTER(c_float)]
        predict.restype = POINTER(c_float)

        set_gpu = star_lib.math21_gpu_set_device_wrapper
        set_gpu.argtypes = [c_int]

        # make_image = star_lib.make_image
        make_image = star_lib.math21_image_create_image_int_input
        make_image.argtypes = [c_int, c_int, c_int]
        # uint32_t
        # make_image.argtypes = [c_int, c_int, c_int]
        make_image.restype = IMAGE

        # self.get_network_boxes = star_lib.math21_ml_net_boxes_get
        # h, w
        self.get_network_boxes = star_lib.math21_ml_function_net_boxes_get
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_int,
                                           POINTER(c_int)]
        self.get_network_boxes.restype = POINTER(DETECTION)

        # self.free_detections = star_lib.free_detections
        self.free_detections = star_lib.math21_ml_function_net_boxes_destroy
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.math21_ml_function_net_create_from_file = star_lib.math21_ml_function_net_create_from_file
        self.math21_ml_function_net_create_from_file.argtypes = [c_char_p, c_char_p, c_int]
        self.math21_ml_function_net_create_from_file.restype = c_void_p

        self.math21_ml_function_net_should_train_continue = star_lib.math21_ml_function_net_should_train_continue
        self.math21_ml_function_net_should_train_continue.argtypes = [c_void_p]
        self.math21_ml_function_net_should_train_continue.restype = c_int

        self.math21_ml_function_net_train_one_mini_batch_in_function = star_lib.math21_ml_function_net_train_one_mini_batch_in_function
        self.math21_ml_function_net_train_one_mini_batch_in_function.argtypes = [c_void_p]
        self.math21_ml_function_net_train_one_mini_batch_in_function.restype = c_float

        self.math21_ml_function_net_get_update_count = star_lib.math21_ml_function_net_get_update_count
        self.math21_ml_function_net_get_update_count.argtypes = [c_void_p]
        self.math21_ml_function_net_get_update_count.restype = c_int64

        self.math21_ml_function_net_set_mbs = star_lib.math21_ml_function_net_set_mbs
        self.math21_ml_function_net_set_mbs.argtypes = [c_void_p, c_int]

        self.math21_ml_function_net_predict_input = star_lib.math21_ml_function_net_predict_input
        self.math21_ml_function_net_predict_input.argtypes = [c_void_p, c_char_p]
        self.math21_ml_function_net_predict_input.restype = c_char_p

        self.math21_ml_function_net_data_feed = star_lib.math21_ml_function_net_data_feed
        self.math21_ml_function_net_data_feed.argtypes = [c_void_p, c_char_p, c_char_p]

        self.math21_ml_function_net_node_log_by_name = star_lib.math21_ml_function_net_node_log_by_name
        self.math21_ml_function_net_node_log_by_name.argtypes = [c_void_p, c_int, c_char_p]

        self.math21_ml_function_net_node_get_data_to_cpu = star_lib.math21_ml_function_net_node_get_data_to_cpu
        self.math21_ml_function_net_node_get_data_to_cpu.argtypes = [c_void_p, c_int, c_char_p]
        self.math21_ml_function_net_node_get_data_to_cpu.restype = c_void_p

        self.math21_ml_function_net_node_get_rawtensor_to_cpu = star_lib.math21_ml_function_net_node_get_rawtensor_to_cpu
        self.math21_ml_function_net_node_get_rawtensor_to_cpu.argtypes = [c_void_p, c_int, c_char_p]
        self.math21_ml_function_net_node_get_rawtensor_to_cpu.restype = m21rawtensor

        self.math21_ml_function_net_save_function_paras = star_lib.math21_ml_function_net_save_function_paras
        self.math21_ml_function_net_save_function_paras.argtypes = [c_void_p, c_char_p]

        self.math21_tensor_2d_float_log_cpu = star_lib.math21_tensor_2d_float_log_cpu
        self.math21_tensor_2d_float_log_cpu.argtypes = [c_char_p, c_char_p, c_int, c_int]

        self.math21_tensor_3d_float_log_cpu = star_lib.math21_tensor_3d_float_log_cpu
        self.math21_tensor_3d_float_log_cpu.argtypes = [c_char_p, c_char_p, c_int, c_int, c_int]

        self.math21_tensor_4d_float_log_cpu = star_lib.math21_tensor_4d_float_log_cpu
        self.math21_tensor_4d_float_log_cpu.argtypes = [c_char_p, c_void_p, c_int, c_int, c_int, c_int]

        self.do_nms_obj = star_lib.math21_ml_box_do_nms_obj
        self.do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_float]

        do_nms_sort = star_lib.math21_ml_box_do_nms_sort
        do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        # self.free_image = star_lib.free_image
        # pointer
        self.free_image = star_lib.math21_image_destroy_image_no_pointer_pass
        self.free_image.argtypes = [IMAGE]

        # star_data_get_labels_from_data_config = star_lib.get_metadata
        self.star_data_get_labels_from_data_config = star_lib.star_data_get_labels_from_data_config
        star_lib.star_data_get_labels_from_data_config.argtypes = [c_char_p]
        star_lib.star_data_get_labels_from_data_config.restype = METADATA

        self.load_image = star_lib.star_image_read_image_normalized_with_rc
        self.load_image.argtypes = [c_char_p, c_int, c_int]
        self.load_image.restype = IMAGE

        # self.predict_image = star_lib.ml_net_predict_image
        self.predict_image = star_lib.math21_ml_function_net_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

        # self.set_font = ImageFont.truetype("extend/msyh.ttf", 12)
        fondFile = "/home/mathxyz/workspace/captcha/extend/msyh.ttf"
        self.set_font = ImageFont.truetype(fondFile, 12)

        self.net = None
        self.meta = None

    def loadFunction(self, function_form, function_paras, data):
        s = time.time()
        self.net = self.math21_ml_function_net_create_from_file(function_form.encode('utf-8'),
                                                                function_paras.encode('utf-8'), 0)
        self.meta = self.star_data_get_labels_from_data_config(data.encode('utf-8'))
        e = time.time()
        print("[load model] speed time: {}s".format(e - s))

    def testFunction(self, data, function_form, function_paras):
        logLevel = 1
        self.star_ml_function_rnn_train(data.encode('utf-8'),
                                        function_form.encode('utf-8'),
                                        # function_paras.encode('utf-8'), 0, logLevel)
                                        None, 0, logLevel)

    @staticmethod
    def sample(probs):
        s = sum(probs)
        probs = [a / s for a in probs]
        r = random.uniform(0, 1)
        for i in range(len(probs)):
            r = r - probs[i]
            if r <= 0:
                return i
        return len(probs) - 1

    @staticmethod
    def c_array(ctype_value, values):
        arr = (ctype_value * len(values))()
        arr[:] = values
        return arr

    def classify(self, net, meta, im):
        out = self.predict_image(net, im)
        res = []
        for i in range(meta.classes):
            res.append((meta.names[i], out[i]))
        res = sorted(res, key=lambda x: -x[1])
        return res

    # 检测指定图片
    def detect(self, image, result_type="box", thresh=.5, hier_thresh=.5, nms=.45):
        """
        box: [
              [b'word', [(59, 105), (93, 105), (93, 140), (59, 140)]],
              [b'word', [(131, 41), (164, 41), (164, 75), (131, 75)]]
             ]
        center: [(b'word', 0.9, (76, 123, 34, 33)),
                 (b'word', 0.9, (148, 58, 33, 33))]
        :param image:
        :param result_type:
        :param thresh:
        :param hier_thresh:
        :param nms:
        :return:
        """
        s = time.time()
        image = image.encode('utf-8')
        img = self.load_image(image, 0, 0)
        num = c_int(0)
        pnum = pointer(num)
        self.predict_image(self.net, img)
        dets = self.get_network_boxes(self.net, img.nr, img.nc, thresh, 0, pnum)

        num = pnum[0]
        if nms:
            self.do_nms_obj(dets, num, nms)

        res = []
        for j in range(num):
            for i in range(self.meta.n):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    res.append((self.meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
        res = sorted(res, key=lambda x: -x[1])
        self.free_image(img)
        self.free_detections(dets, num)

        e = time.time()
        print("[detect image - i] speed time: {}s".format(e - s))

        if result_type == "center":
            # 这里返回的是中心点加上边距的坐标
            new_res = list()
            for res_ele in res:
                res_ele = list(res_ele)
                if isinstance(res_ele[0], bytes):
                    res_ele[0] = res_ele[0].decode("utf-8")
                new_res.append(res_ele)
            return new_res
        else:
            # 这里返回的是box四个点的坐标
            boxes = self.calculation_boxes(res)
            return boxes

    # 将中心距的结果计算为box的结果
    @staticmethod
    def calculation_boxes(res):
        result = []
        labels = []
        for inf in res:
            label = inf[0]
            if isinstance(label, bytes):
                label = label.decode("utf-8")
            labels.append(label)

            location = inf[2]
            result.append(location)

        boxes = []
        for index, r in enumerate(result):
            cx, cy, w, h = r
            a = cx - (h / 2)
            c = cx + (h / 2)
            b = cy - (w / 2)
            d = cy + (w / 2)
            box = [labels[index],
                   [(a, b), (c, b), (c, d), (a, d)]]
            boxes.append(box)

        return boxes

    def cut_and_save(self, filename, save_path):
        base_path = os.path.join(save_path, "crop_result")
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        res = self.detect(filename, result_type="center")

        result = []
        labels = []
        for r in res:
            label = r[0]
            if isinstance(label, bytes):
                label = label.decode("utf-8")
            labels.append(label)
            location = r[2]
            result.append(location)

        img = Image.open(filename)
        for index, r in enumerate(result):
            cx, cy, w, h = r
            a = cx - (h / 2)
            c = cx + (h / 2)
            b = cy - (w / 2)
            d = cy + (w / 2)
            cropedimage = img.crop((a, b, c, d))
            cropedimage.save(os.path.join(save_path, "crop_result/{}_{}.jpg".format(labels[index], index)))

    # 在图像上面画box
    def draw_boxes(self, filename, boxes):
        img = Image.open(filename)
        draw = ImageDraw.Draw(img)
        for box_label in boxes:
            # 解包
            label = box_label[0]
            box = box_label[1]
            # 计算文字位置
            x, y = box[0]
            y = y - 15
            # 添加文字
            draw.text((x, y), label, font=self.set_font, fill=(0, 0, 0))
            # 添加盒子的边界
            draw.line([box[0], box[3]], fill="red")
            draw.line([box[3], box[2]], fill="red")
            draw.line([box[2], box[1]], fill="red")
            draw.line([box[1], box[0]], fill="red")

        return img

    # 展示检测后的图片
    def show_and_save(self, filename):
        result = self.detect(filename, result_type="box")
        img = self.draw_boxes(filename, result)
        plt.imshow(img)
        plt.show()
        img.save("text.jpg")

    # 保存带有box的图片
    def save(self, filename):
        result = self.detect(filename, result_type="box")
        img = self.draw_boxes(filename, result)
        img.save("text.jpg")


if __name__ == '__main__':
    mr = Math21Recognize(
        function_form="app/my_captcha/my_captcha_train.yolov3.cfg",
        function_paras="app/my_captcha/backup/my_captcha_train.backup",
        data="app/my_captcha/my_captcha.data"
    )
    mr.show_and_save("app/my_captcha/images_data/JPEGImages/0_15463993913409665.jpg")
    mr.cut_and_save("app/my_captcha/images_data/JPEGImages/0_15463993913409665.jpg", "crop_test")
    # rv = dr.detect("app/my_captcha/images_data/JPEGImages/0_15463993913409665.jpg")
    # print(rv)
