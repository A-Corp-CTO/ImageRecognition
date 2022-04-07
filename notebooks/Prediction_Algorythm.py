
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score



def cancer_model(image, mask, model = tree_trained_model, acc_score = tree_acc_score,
                 auc_score = tree_auc_score, f1_score = tree_f1_score):
    """Given the an RGB JPG colour image and a PNG binary mask
    of a skin lesion this function will print the predicted
    diagnosis along with an evaluation of the model.
    Also returns the diagnosis as a float."""
    
    image_crop = image.crop(mask.getbbox())
    mask_crop = mask.crop(mask.getbbox())
    
    # Make the length and height of the the image and mask even 
    image_crop = make_sides_even(image_crop)
    cropped_mask = make_sides_even(mask_crop)

    # Instantiate a blank image for a composite image
    tmp_image = Image.new("RGB",image_crop.size, 0)

    # Create a composite image based on the blank image, 
    # the cropped image and the mask 
    filtered_image = Image.composite(image_crop,tmp_image,mask_crop)
    
    
    feature_dictionary = {
        "perimeter": [], 
        "asymmetry": [], 
        "red_average": [],
    }
    
    area, perimeter = get_area_perimeter(cropped_mask)
    feature_dictionary['perimeter'].append(perimeter)
    feature_dictionary['asymmetry'].append(get_asymmetry(cropped_mask))
    red, green, blue = get_avg_color(filtered_image)
    feature_dictionary['red_average'].append(red)
    features = pd.DataFrame(feature_dictionary)
    prediction = model.predict(features)
    if int(prediction) == 1:
        diagnosis = "Melanoma"
    elif int(prediction) == 0:
        diagnosis = "Healthy"
    print("The predicition is ", diagnosis, " with probability unknown.\n"
          "The accuracy score of the model is ", acc_score, ".\n"
          "The ROC AUC score of the model is ", auc_score, ".\n"
          "The f1 score of the model is ", f1_score, ".\n"
          "The model does not provide a reliable diagnosis"
         )
    return None



if __name__ == "__main__":
    
    features = pd.read_csv("../features/feature_set.csv", sep=";", index_col=False)
    feature_list = ["perimeter", "asymmetry", "red_average"]
    X_main = features[feature_list]

    # Creates a random variable between 1 and 100 to facilitate splitting the data into multiple sets
    np.random.seed(0)
    separator = np.random.randint(1, 101, size = (image_data.shape[0], 1))

    X_scaled = StandardScaler().fit_transform(X_main.values)
    X_scaled_df = pd.DataFrame(X_scaled, index=X_main.index, columns=X_main.columns)
    X_scaled_df["separator"] = separator

    y = image_data["melanoma"]

    X_train, X_val, y_train, y_val = train_test_split(X_main, y, stratify=y)

    tree_trained_model = DecisionTreeClassifier().fit(X_train, y_train)
    prediction = tree_trained_model.predict(X_val)
    tree_acc_score = accuracy_score(y_val, prediction)
    tree_auc_score = roc_auc_score(y_val, prediction)
    tree_f1_score = f1_score(y_val, prediction)




