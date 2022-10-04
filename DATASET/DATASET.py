import colorsys
from PIL import Image  # 利用image模块
# 输入图片名称如下，
filename = '1.jpg'#将所需要改变颜色的图片和代码放入相同文件夹，文件格式可以是jpg、tif、...
# 读入图片，转化为 RGB 色值
image = Image.open(filename).convert('RGB')
# print("RGB值为", image.split())
# 将 RGB 色值分离
image.load()
r, g, b = image.split()
result_r, result_g, result_b = [], [], []
# 依次对每个像素点进行处理
for pixel_r, pixel_g, pixel_b in zip(r.getdata(), g.getdata(), b.getdata()):
    R = 0.393 * pixel_r + 0.769 * pixel_g + 0.289 * pixel_b
    G = 0.349 * pixel_r + 0.686 * pixel_g + 0.168 * pixel_b
    B = 0.272 * pixel_r + 0.534 * pixel_g + 0.131 * pixel_b
    R = (R + pixel_r) / 2
    G = (G + pixel_g) / 2
    B = (B + pixel_b) / 2
    # 每个像素点结果保存
    result_r.append(R)
    result_g.append(G)
    result_b.append(B)

r.putdata(result_r)
g.putdata(result_g)
b.putdata(result_b)

# 合并图片
image = Image.merge('RGB', (r, g, b))
image.save("3.jpg")#保存修改像素点后的图片，名称可以自行修改

