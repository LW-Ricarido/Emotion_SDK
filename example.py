from Emotion_Recognize import Emotion_Recognize
import torch
import your_dataset

# image emotion example 图片表情样例
image_examples = Emotion_Recognize(Video=False,pretrain=False,nGPU=0)
# Prepare dataset 准备数据集
Img_train_loader = your_dataset.get_Img_train_loader(batch_size=32,workers=0)
Img_test_loader = your_dataset.get_Img_test_loader(batch_size=32,workers=0)
Img_recog_loader = your_dataset.get_Img_recog_loader(batch_size=32,workers=0)
#Img_recog_loader = your_dataset.get_Img_recog_loader(batch_size=32,workers=0)
# train your own model 训练你自己的模型
# train_summary = image_examples.train(Img_train_loader,Img_test_loader,epochs=1)
# test your own model 测试你自己的模型
# test_summary = image_examples.eval(Img_test_loader)
# recognize single image 识别单张图像
# output = image_examples.recognize(torch.rand(1,3,100,100))
# recognize multi images 识别多张图像
#output_dict = image_examples.batch_recognize(Img_recog_loader)



# Video emotion example   视频表情样例
video_example = Emotion_Recognize(Video=True,pretrain=False,nGPU=0)
# Prepare dataset 准备数据集
Video_train_loader = your_dataset.get_Video_train_loader(batch_size=2,workers=0)
Video_test_loader = your_dataset.get_Video_test_loader(batch_size=2,workers=0)
Video_recog_loader = your_dataset.get_Video_recog_loader(batch_size=2,workers=0)
# train your own model 训练你自己的模型
#train_summary = video_example.train(Video_train_loader,Video_test_loader,epochs=1)
# test your own model 测试你自己的模型
#test_summary = video_example.eval(Video_test_loader)
# recognize single image 识别单张图像
#output = video_example.recognize(torch.rand(1,3,16,224,224))
# recognize multi images 识别多张图像
output_dict = video_example.batch_recognize(Video_recog_loader)

