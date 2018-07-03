from train import *

model = None
optimizer = None
inputargs = None

def main():
    global inputargs
    
    print('Hi! Welcome to the Predict script')
    
    # Collect args and parse them
    parser = argparse.ArgumentParser(description='Arguments to load the prediction script. USAGE: predict.py input checkpoint --top_k')

    #predict.py input checkpoint --top_k 3
    parser.add_argument('input', action="store", type=str, help='Path to the input image (required)')
    parser.add_argument('checkpoint', action="store", type=str, help='Model training checkpoint (required)')
    parser.add_argument('--top_k', action="store", type=int, default=5, help='The number of predictions to return for each input (default = 5)')
    
    inputargs = parser.parse_args()
    
    model, optimizer = load_checkpoint(inputargs.checkpoint)
    
    print('Loaded model with the following keys:')
    print(model.state_dict().keys())

    # Run the prediction engine
    probs, classes = predict(inputargs.input, model)

    # Get the labels
    labels = []
    for i in range(len(classes)):
        labels.append(cat_to_name[classes[i]])

    print('File path ' + inputargs.input)
    print('\nMost likely classification: ' + (labels[0]).title() + ' (' + str(round(probs[0]*100)) + '%)\n')

    # Display the chart
    currentflower = pd.DataFrame({'Probability': probs, 'Flower Classification': labels})
    currentflower.set_index('Flower Classification')
    #sb.barplot(data=currentflower, x = 'Probability', y= 'Flower Classification')

    # Display the image
    #image = Image.open(image_path)
    #processed_image = process_image(image)
    #imshow(processed_image)
    
    print('Full table:')
    print(currentflower)
    
    
def load_checkpoint(filepath):
    global optimizer
    global model
    
    checkpoint = torch.load(filepath)
    
    arch = checkpoint['arch']
    model = getattr(models, arch)(pretrained=True)

    model.classifier = None
    model.classifier = checkpoint['classifier']
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #Resize the image
    size = 256, 256
    image = image.resize(size)
    
    #Define the center box and crop
    boxlen = (256 - 224) / 2
    box = boxlen, boxlen, 256 - boxlen, 256 - boxlen
    image = image.crop(box)
    
    #Colour adjustment
    np_image = np.array(image)
    np_image = np_image / 255
    
    #Do image normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    np_image = np_image.transpose((2, 0, 1))
    
    return (np_image)


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# DONE: Implement the code to predict the class from an image file
def predict(image_path, model):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    topk = inputargs.top_k
    
    cuda = torch.cuda.is_available()
    if cuda:
        model.to("cuda:0")
        device="cuda"
        #print('Using Cuda')
    else:
        model.to("cpu")
        device="cpu"
        #print('Using CPU')


    model.eval()  # Put the model in evaluation mode

    image = Image.open(image_path)
    processed_image = process_image(image)
    processed_image = torch.from_numpy(np.array([processed_image])).float() # Convert to Tensor
    
    inputs = Variable(processed_image)
    
    # Move to the GPU
    if cuda:
        inputs = inputs.cuda()


    outputs = model.forward(inputs)
    ps = torch.exp(outputs)
    
    torchreturn = torch.topk(ps, topk)
    probs = torchreturn[0]
    indexes = torchreturn[1]
    
    # Extract the values we need
    probs = probs.tolist()[0]
    indexes = indexes.tolist()[0]
     
    conersionindexes = []
    for i in range(len(model.class_to_idx.items())):
        conersionindexes.append(list(model.class_to_idx.items())[i][0])

    classes = []
    for i in range(topk):
        classes.append(conersionindexes[indexes[i]])
        
    return probs, classes

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
# Call to main function to run the program
if __name__ == "__main__":
    main()

