from fileinput import filename
import requests
import os
if not os.path.exists('./验证码'):
    os.mkdir('./验证码')
headers={
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.0.0 Safari/537.36'
}
url = 'https://jaccount.sjtu.edu.cn/jaccount/captcha?uuid=e956d943-49c2-45f4-991d-aef34605c890&t=1657440018672'

for i in range(1,5):
    img_data = requests.get(url=url,headers=headers).content
    filename = str(i)+'.jpg'
    filepath = '验证码/'+filename
    with open(filepath,'wb') as fp:
        fp.write(img_data)
print('完毕')

