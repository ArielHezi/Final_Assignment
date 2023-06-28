from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from datetime import datetime
from sklearn.impute import KNNImputer
import numpy as np
import warnings
warnings.filterwarnings("ignore")

###### סידור דאטה
def prepare_data(train_data):
    train_data.columns = train_data.columns.str.replace(' ', '')
    def fix_price(string):
        try:
            numbers = []
            num = ""
            ok=0
            for char in str(string):
                if char.isnumeric():
                    ok=1
                    num += char
                    numbers.append(num)
                    continue
                elif char == ',': 
                    continue 
                elif char == '.': 
                    num += char
                    continue 
                else: 
                    numbers.append(num)
                    num = ""
            
            if len(numbers) == 0:
                return None
            elif ok==0:
                return None
            else:
                if len(set(str(max(numbers)))) == 1:
                    return None
                return max(numbers)
        except:
            return None

    def fix_street(string):
        try:
            String = ""
            
            if pd.isnull(string):  # Check for missing values
                return string

            if isinstance(string, list):
                String = str(string)
            else:
                String = str(string)

            # Remove single quotes at the beginning and end of the string
            String = re.sub(r"'^\s*|\s*'$", '', String)

            # Remove all remaining single quotes in the string
            String = String.replace("'", '')

            if "(" in String or ")" in String:
                String = String.replace("(", "").replace(")", "").strip()
            elif "[" in String or "]" in String:
                String = String.replace("[", "").replace("]", "").strip()
            elif "/" in String:
                names = String.split("/")
                String = names[-1]
            elif '"' in String:
                String = re.sub(r'"', '', String)
            elif "רמב ן" in String:
                return "רמבן"
            else:
                result = ""
                for char in String:
                    if not (str(char).isnumeric() and int(char) < 1000):
                        result += char
            String = String.strip()

            return String
        except:
            return String

    def fix_Area(string):
        try:
            string=str(string)
            num_list = string.split()
            for i in num_list:
                if i.isdigit():
                    return int(i)
            else:
                return None
        except:
            return None

    def fix_room_number(text):
        try:
            if pd.isnull(text):
                return None
            numbers = []
            num = ""
            for char in str(text):
                if char.isnumeric() or char == '.':
                    num += char
                    continue
                elif char == ',':
                    continue
                else:
                    if num != "":
                        numbers.append(float(num))
                        num = ""
            if num != "":
                numbers.append(float(num))
            if len(numbers) == 0:
                return None
            return max(numbers)
        except:
            return None

    def fix_number_in_street(num):
        try:
            if num>1000:
                return None
            else:
                return num
        except:
            return num

    def fix_city_area(string):
        try:
            if isinstance(string, float):
                return None
            elif "-" in string or "תוכנית" in string:
                return None
            else:
                return string
        except:
            return None


    def fix_furniture(string):
        try:
            if ("חלקי" in string) or ("מלא" in string) or ("יש" in string) or (string==True) or (string=="TRUE") or ("yes" in string) or ("Yes" in string) or ("כן" in string):
                return 1
            else: 
                 return 0  
        except:
            return 0

    def fix_category(string):
        try:
            if ("אין" in string) or (string==False) or (string=="FALSE") or ("no" in string) or ("No" in string) or ("לא" in string):
                return 0
            elif ("יש" in string) or (string==True) or (string=="TRUE") or ("yes" in string) or ("Yes" in string) or ("כן" in string):
                return 1
            else:
                return None
        except:
            return None

    def fix_category_handicapFriendly(string):
        try:
            if ("נגיש" in string) or ("יש" in string) or (string==True) or (string=="TRUE") or ("yes" in string) or ("Yes" in string) or ("כן" in string):
                return 1
            elif ("אין" in string) or (string==False) or (string=="FALSE") or ("no" in string) or ("No" in string) or ("לא" in string):
                return 0
            else:
                return None
        except:
            return None


    def fix_condition(string):
        try:
            if ("לא צויין" in string) or ("ישן" in string) or (string==False) or (string=="FALSE"):
                return "דורש שיפוץ"
            elif ("חדש" in string) or ("1" in string) or (string==True) or (string=="TRUE") or ("yes" in string) or ("Yes" in string) or ("כן" in string):
                 return "חדש"
            elif ("renovated" in string) or ("משופץ" in string):
                 return "משופץ"
            elif ("שמור" in string) or ("SAVE" in string) or ("Save" in string):
                 return "שמור"
            else:
                return "דורש שיפוץ"
        except:
            return "דורש שיפוץ"    


    def give_floor(string):
        try:
            lst = string.split()
            if len(lst) == 4:
                if lst[1] == "קרקע":
                    return 1
                if lst[1].isdigit:
                    return lst[1]
            elif len(lst) == 2:
                if lst[0] == "קומת" and lst[1] == "קרקע":
                    return 1
                elif lst[0] == "קומת" and lst[1] == "מרתף":
                    return 0
                elif lst[0] == "קומה" and lst[1].isdigit():
                    return lst[1]
            elif len(lst) == 1:
                if lst[0] == "קרקע":
                    return 1
                elif lst[0].isdigit():
                    return lst[0]

                elif "-" in lst[0]:
                    return None           
            else:
                return None
        except:
            return None


    def give_total_floor(string):
        try:
            lst = string.split()
            if len(lst) == 4:
                if lst[1] == "קרקע":
                    return lst[3]
                elif lst[3].isdigit:
                    return lst[3]

            elif len(lst) == 2:
                if lst[0] == "קומת" and lst[1] == "קרקע":
                    return 1

                elif lst[0] == "קומת" and lst[1] == "מרתף":
                    return 0

                elif lst[0] == "קומה" and lst[1].isdigit():
                    return None

            elif len(lst) == 1:
                if lst[0] == "קרקע":
                    return 1

                elif lst[0].isdigit():
                    return None

                elif "-" in lst[0]:
                    return None
            else:
                return None    
        except:
            return None


    def fix_num_images(string):
        try:
            if isinstance(string, int) or isinstance(string ,float):
                integer = int(string)
                return integer
            else:
                digits = re.findall(r'\d+', string)
                integer = int(''.join(digits))
                return integer
        except:
            return None


    def fix_published_days(string):
        try:
            digits = re.findall(r'\d+', string)
            if digits:
                integer = int(''.join(digits))
                return integer
        except:
            return None


    def convert_entrance_date(string):
        try:
            if "לא צויין" in string:
                return 'not_defined'
            elif "גמיש" in string:
                return "flexible"
            elif "מיידי" in string:
                return "less_than_6_months"
            else:
                return string
        except:
            return string


    def entrance_date_to_months(date):
        try:
            current_date = datetime.now().date()
            months_diff = (date.year - current_date.year) * 12 + (date.month - current_date.month)
            return max(months_diff, 0)
        except AttributeError:
            return date
    
    def fix_entrance_date(string):
        try:
            if string < 6:
                return 'less_than_6_months'

            elif 6 <= string <= 12:
                return 'months_6_12'

            elif string > 12:
                return 'above_year'   
        except:
            return string

    def fix_City(string):
        try:
            if 'נהריה' in string or 'נהרייה' in string:
                return 'נהריה'
            else: 
                return string
        except:
            return string


    def fix_type(string):
        try:
            if "בניין" in string:
                return 'דירה'
            elif "נחלה" in string:
                return 'מגרש'
            elif "קוטג'" in string:
                return 'קוטג'

            elif "קוטג' טורי" in string:
                return 'קוטג'

            else: 
                return string
        except:
            return string
    
    def check_keywords(text):
        num=100
        try:
            keywords = ["אדריכלי","יוקרתי","שדות","מפואר","אור","חצר","מיקום","חדר כושר","פארק","בריכה","בריכת","גדול","יוקרתי","גינה" "יחידת הורים","נוף" ,"ים","מוטפחת","קרוב"]
            for keyword in keywords:
                if keyword in text:
                    num = num+20
                    
            keywords = ["ארנונה","ילדיים","חינוך","להשקעה","מתווכים" ,"פינוי בינוי","תמא","ללא תיווך","מושכרת","מחולקת","חרדי","משקיעים","מפוצלת","תשואה","לא בשבת","אוטובוס","טובה","כנסת","פוטנציאל","בתי","מקרר","ריצוף","יחידות","קטן"]
            for keyword in keywords:
                if keyword in text:
                    num = num-20
            return num
        except:
            return num
        
    def fill_nan_with_values(df, col1, col2):
        try:
            mask1 = df[col1].isna() & df[col2].notna()
            mask2 = df[col2].isna() & df[col1].notna()

            df.loc[mask1, col1] = df.loc[mask1, col2]
            df.loc[mask2, col2] = df.loc[mask2, col1]
            
            df = df.dropna(subset=[col1, col2]) 
            return df
        except:
            return None

    
    # Fill missing values in both columns
    train_data = fill_nan_with_values(train_data, 'Street', 'city_area')
    
    
    #fix price
    train_data['price'] = train_data['price'].apply(fix_price)

    #fix City
    train_data['City'] = train_data['City'].apply(fix_City)

    #fix type
    train_data['type'] = train_data['type'].apply(fix_type)

    #fix street
    train_data['Street'] = train_data['Street'].apply(fix_street)

    #fix area
    train_data['Area'] = train_data['Area'].apply(fix_Area)

    #fix room_number
    train_data['room_number'] = train_data['room_number'].apply(fix_room_number)

    #fix number_in_street
    train_data['number_in_street'] = train_data['number_in_street'].apply(fix_room_number)
    train_data['number_in_street'] = train_data['number_in_street'].apply(fix_number_in_street)

    #fix city_area
    train_data['city_area'] = train_data['city_area'].apply(fix_street)
    train_data['city_area'] = train_data['city_area'].apply(fix_city_area)

    #fix condition
    train_data['condition'] = train_data['condition'].apply(fix_condition)

    #fix floor_out_off - (floor and total floors)
    ##Adding a floor column
    train_data.insert(9, 'floor', train_data['floor_out_of'])
    train_data['floor'] = train_data['floor'].apply(give_floor)

    ##Adding a total_floors column
    train_data.insert(10, 'total_floors', train_data['floor_out_of'])
    train_data['total_floors'] = train_data['total_floors'].apply(give_total_floor)

    #fix num_of_images
    train_data['num_of_images'] = train_data['num_of_images'].apply(fix_num_images)

    #fix publishedDays
    train_data['publishedDays'] = train_data['publishedDays'].apply(fix_published_days)
    
    #fix description
    train_data['description'] = train_data['description'].apply(check_keywords)

    #fix entrance_date
    train_data['entranceDate']= train_data['entranceDate'].apply(convert_entrance_date).apply(entrance_date_to_months).apply(fix_entrance_date)
    #change to boolean
    train_data['hasElevator'] = train_data['hasElevator'].apply(fix_category)
    train_data['hasParking'] = train_data['hasParking'].apply(fix_category)
    train_data['hasBars'] = train_data['hasBars'].apply(fix_category)
    train_data['hasStorage'] = train_data['hasStorage'].apply(fix_category)
    train_data['hasAirCondition'] = train_data['hasAirCondition'].apply(fix_category)
    train_data['hasBalcony'] = train_data['hasBalcony'].apply(fix_category)
    train_data['hasMamad'] = train_data['hasMamad'].apply(fix_category)
    train_data['handicapFriendly'] = train_data['handicapFriendly'].apply(fix_category_handicapFriendly)
    train_data['furniture'] = train_data['furniture'].apply(fix_furniture)

    #delete properties without price
    train_data.dropna(subset=['price'], inplace=True)

    train_data = train_data.drop_duplicates() #מחיקת רשומות כפולות
    
    train_data = train_data.drop(['floor_out_of'], axis=1)
    train_data = train_data[train_data["type"] != 'אחר']
    train_data = train_data[train_data["room_number"] != 35]
    
    ### מילוי ערכים חסרים בשטח ובמספר חדרים
    df = train_data
    df = df.drop(['type', 'City', 'Street', 'number_in_street','description', 'city_area','condition' ,'furniture', 'num_of_images', 'publishedDays', 'entranceDate','floor','total_floors','hasElevator','hasParking','hasBars','hasStorage','hasAirCondition','hasBalcony','hasMamad','handicapFriendly'], axis=1)
    # create an object for KNNImputer
    imputer = KNNImputer(n_neighbors = 5)
    After_imputation = imputer.fit_transform(df)
    # Replace the missing values in train_data with the imputed values
    train_data.loc[:, df.columns] = np.round(After_imputation * 2) / 2

    
    ###  מכן התחלנו למלא ערכים ריקים בעמודות שנשארו
    df = train_data
    df = df.drop(['condition','entranceDate','publishedDays','type', 'City','description', 'Street','number_in_street', 'city_area'], axis=1)
    # create an object for KNNImputer
    imputer = KNNImputer(n_neighbors= 1)
    After_imputation = imputer.fit_transform(df)
    # Replace the missing values in train_data with the imputed values
    train_data.loc[:, df.columns] = np.round(After_imputation * 2) / 2

    train_data = train_data.drop(['number_in_street', 'publishedDays','num_of_images','hasAirCondition','handicapFriendly','furniture','condition'], axis=1)    
    # Create a copy of the train_data to remove outliers
    df1 = train_data.copy()

    # Define the features to handle outliers
    features1 = df1.columns  # Replace [...] with your desired feature list

    # Remove outliers using IQR method
    for i in features1:
        if np.issubdtype(df1[i].dtype, np.number):  # Check if column contains numeric data
            Q1 = df1[i].quantile(0.25)
            Q3 = df1[i].quantile(0.75)
            IQR = Q3 - Q1
            df1 = df1[(df1[i] >= (Q1 - 1 * IQR)) & (df1[i] <= (Q3 + 1 * IQR))]
            df1 = df1.reset_index(drop=True)

    train_data = df1

    return (train_data)
