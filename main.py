import pandas as pd

df = pd.read_csv('fashion_again_.csv')
ds=pd.read_csv('label_encoded_dataset.csv')
import streamlit as st
st.title("Smart Clothing Recommendation System")
# prompt: INPUT gender and apparel typ
gender = st.selectbox("Select Gender", ["Male", "Female"])
if gender=="Male":
  gender="Men's"
  gender_no=0
elif gender=="Female":
  gender="Women's"
  gender_no=1

apparel = st.selectbox("Select apparel", ["All-body", "Bottomwear","Outerwear","Topwear"])

#apparel=input('Enter your apparel: (the number in brakets)')
if apparel=='All-body':
  apparel_type='0'
elif apparel=='Bottomwear':
  apparel_type='1'
elif apparel=='Outerwear':
  apparel_type='2'
elif apparel=='Topwear':
  apparel_type='3'

new_ds=ds[ds['Category']==int(gender_no) ]
new_ds=new_ds[new_ds['Apparel']==int(apparel_type)]
new_df=df[(df['Category']==str(gender))]
new_df=new_df[new_df['Apparel']==str(apparel)]
unique_sub_category=new_df['sub-category'].unique()
print(unique_sub_category)
#new_df
import json

with open('label_mappings.json', 'r') as file:
    label_mappings = json.load(file)

clothes_ = []

for key, value in label_mappings['sub-category'].items():
    if key in unique_sub_category:
        clothes_.append(key)
        #clothes = st.checkbox(key)
if gender is not None:
    if apparel is not None:
       clothes=st.selectbox("Select sub-category",clothes_)

for key, value in label_mappings['sub-category'].items():
    if key == clothes:
        cloth=value
        
new_ds= new_ds[new_ds['sub-category'] == int(cloth)]

new_df=new_df[new_df['sub-category']==str(clothes)]
unique_occasions = new_df['Occasion'].unique()
ocassion=[]
for key, value in label_mappings['Occasion'].items():
   if key in unique_occasions:
        ocassion.append(key)

if gender is not None:
    if apparel is not None:
       if ocassion is not None:
          occasion=st.selectbox("Select Occasion",ocassion)

for key, value in label_mappings['Occasion'].items():
    if key == occasion:
        occasion_number=value


if gender=="Women's":
  label= {
    "Apple": 0,
    "Curvy": 3,
    "Hourglass": 4,
    "Pear": 7,
    "Petite": 8,
    "Rectangle": 9,
    "Slim": 11,
    "Tall": 12
  }
if gender=="Men's":
   label={
      "Athletic":1,
      "Broad shoulders":2,
      "Lean":5,
      "Muscular":6,
      "Regular":10,
      "Slim":11,
      "Tall":12
   }
selected_number = None

body_type = st.selectbox("Select your Body Type", list(label.keys()))
body_type_number = label[body_type]

season_mapping = {
        "Fall": 0,
        "Spring": 1,
        "Summer": 2,
        "Winter": 3
    }
    
    # Create selectbox for choosing season
selected_season = st.selectbox("Select Season", list(season_mapping.keys()))
    
    # Save the respective number in a different variable
season_number = season_mapping[selected_season]

submit_button = st.button("Submit")


new_ds=new_ds[(new_ds['Apparel']==int(apparel_type)) ]
new_ds=new_ds[(new_ds['Season']==int(season_number)) |
                new_ds['Body Type Suitability']==int(body_type_number) |
                (new_ds['Occasion']==int(occasion_number))
                ]

from model import model_running
if submit_button:

  
  type1,fit,pattern,material,Neckline=model_running(new_ds,gender_no,cloth,apparel_type,body_type_number,season_number,occasion_number)
  # prompt: from label_mappings.json print key where value ==predicted_type




  # Import necessary libraries
  from bs4 import BeautifulSoup
  import requests

  # Define your search query
  if Neckline=='Unknown':
    search_query =str(gender)+str(pattern)+str(material)+str(fit)+str(type1)
  elif ((fit=='Oversized') & (type1=='Oversized T-Shirt')):
    search_query =str(gender)+str(Neckline)+str(pattern)+str(material)+str(type1)
  else:
    search_query =str(gender)+str(Neckline)+str(pattern)+str(material)+str(fit)+str(type1)
  # Define the desired image size (e.g., 'large', 'medium', 'icon', etc.)
  image_size = 'large'

  # Perform the Google Images search with specified image size
  url = f"https://www.google.com/search?q={search_query}&tbm=isch&s={image_size}"
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'html.parser')

  # Extract the URLs of the first five images from the search results
  image_urls = []
  for img in soup.find_all('img', class_="DS1iW")[:5]:  # Limit to the first 5 images
      image_url = img.get('src')
      if image_url:
          image_urls.append(image_url)

  import streamlit as st
  import requests
  from PIL import Image
  from io import BytesIO
  col=[]
  col = st.columns(5)
  # Assuming image_urls is a list of image URLs
  for i, image_url in enumerate(image_urls):
      # Get the image from the URL
      response = requests.get(image_url)

      # Check if the response is successful
      if response.status_code == 200:
          # Display the image
          with col[i]:
            img = Image.open(BytesIO(response.content))
            st.header(i+1)
            st.image(img)

      else:
          st.error(f"Failed to retrieve image from URL: {image_url}")

