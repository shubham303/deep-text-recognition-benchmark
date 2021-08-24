#!/usr/bin/env python
# coding: utf-8

# In[ ]:


output = open("log_demo_result.txt").readlines()

# In[ ]:


from IPython.core.display import display, HTML
from PIL import Image
import base64
import io
import pandas as pd

data = pd.DataFrame()
for ind, row in enumerate(output[output.index('image_path               	predicted_labels         	confidence '
                                              'score\n')+2:]):
  if ind < 50:
    row = row.split('\t')
    print(row)
    filename = row[0].strip()
    label = row[1].strip()
    conf = row[2].strip()
    img = Image.open(filename)
    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    imgStr = base64.b64encode(img_buffer.getvalue()).decode("utf-8") 

    data.loc[ind, 'img'] = '<img src="data:image/png;base64,{0:s}">'.format(imgStr)
    data.loc[ind, 'id'] = filename
    data.loc[ind, 'label'] = label
    data.loc[ind, 'conf'] = conf
  else:
    break

html_all = data.to_html(escape=False)
display(HTML(html_all))


# In[ ]:


type(output)


# In[ ]:


import pandas as pd
l = []
for ind, row in enumerate(output[3:]):
    try:
        row = row.split('\t')
        print(row)
        filename = row[0].strip()
        label = row[1].strip()
        conf = row[2].strip()
        l.append([filename, label, conf])
    except:
      continue

data = pd.DataFrame(l, columns=['filename', 'prediction', 'confidence'])


# In[ ]:


data.to_csv("Prediction_English_FT.csv", index=False, header=True)


# In[ ]:


len(data)


# In[ ]:




