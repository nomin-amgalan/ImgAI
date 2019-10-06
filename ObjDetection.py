# Using the Google Cloud Vision AI we detect the objects and its coordinates

# Function for detecting objects/texts and sending them out as a collection



from google.cloud import vision
import io
import os


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ImgAI.json"

client = vision.ImageAnnotatorClient()


def detect_objects(file_path):

    # SET UP
    with io.open(file_path, 'rb') as img_file:
        content = img_file.read()
    image = vision.types.Image(content = content)

    # LOCATIONS
    landmark_response= client.landmark_detection(image = image)
    landmarks = landmark_response.landmark_annotations
    #for landmark in landmarks:
        #print(landmark.description)

    # LOGOS
    logo_response = client.logo_detection(image=image)
    logos = logo_response.logo_annotations

    # OBJECTS
    objects = client.object_localization(image=image).localized_object_annotations

    # PRINT RESULTS
    if len(landmark_response.landmark_annotations) != 0:
        print("Landmarks:")
        for landmark in landmarks:
            print(landmark.description)
    if len(logo_response.logo_annotations) != 0:
        print("Logos:")
        for logo in logos:
            print(logo.description) 
    if len(objects) != 0:
        print("Other Objects:")
        for obj in objects:
            print(obj.name)

    return landmarks, logos, objects


detect_objects("bigben.jpeg")

