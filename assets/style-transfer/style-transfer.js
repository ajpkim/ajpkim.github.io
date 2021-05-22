const model = new mi.ArbitraryStyleTransferNetwork();

const contentImage = document.getElementById('content-image');
const styleImage = document.getElementsByClassName('style-image')[0];
const canvas = document.getElementById('canvas');
const addStyleImageBtn = document.getElementById('add-style-image-btn');
const styleTransferBtn = document.getElementById('style-transfer-btn');
const fileUploader = document.getElementById('file-upload');
const styleContainer = document.getElementById('style-container');
const styleImageBlock = document.getElementById('style-image-block-1');
const selectStyleList = document.getElementsByClassName('style-select')[0];
const selectContentList = document.getElementById('content-select');
const saveStylizedBtn  = document.getElementById('save-stylized-image-btn');

let styleImageCount = 1
//////////////////////////////////////////////////////////////////////
selectContentList.addEventListener('change', () => {
    uploadImage(selectContentList, 'content');
});

selectStyleList.addEventListener('change', () => {
    uploadImage(selectStyleList, 'style');
});

addStyleImageBtn.addEventListener('click', () => {
    addNewStyleBlock();
});

styleTransferBtn.addEventListener('click', () => {
    stylize();
});

saveStylizedBtn.addEventListener('click', () => {
    let downloadLink = document.createElement('a');
    downloadLink.setAttribute('download', 'stylized.png');
    canvas.toBlob(function(blob) {
	let url = URL.createObjectURL(blob);
	downloadLink.setAttribute('href', url);
	downloadLink.click();
    });
});

fileUploader.addEventListener('change', () => {
    if (fileUploader.styleOrContent === 'content') {
	contentImage.src = URL.createObjectURL(fileUploader.files[0]);
    } else {
	fileUploader.styleImage.src = URL.createObjectURL(fileUploader.files[0]);
    }
});

function getStyleImages() {
    return Array.from(document.getElementsByClassName('style-image'));
}

function getStyleWeights() {
    let styleWeights = document.getElementsByClassName('style-slider');
    return Array.from(styleWeights).map(slider => Number(slider.value));
}

function getContentStyleRatio(styleWeights) {
    let total = styleWeights.length * 10;
    return 1 - (styleWeights.reduce((a, b) => a + b) / total)
}

function addNewStyleBlock() {
    styleImageCount += 1;
    let block = styleImageBlock.cloneNode(true);
    block.id = `style-image-block-${styleImageCount}`;

    let img = block.querySelector('.style-image');
    img.id = `style-image-${styleImageCount}`;

    let styleSelectionList = block.querySelector('.style-select');
    styleSelectionList.addEventListener('change', () => {
	uploadImage(styleSelectionList, 'style', img);
    });

    let removeBtn = block.querySelector('.remove-style-btn');
    removeBtn.id = `remove-style-btn-${styleImageCount}`;
    removeBtn.addEventListener('click', () => block.remove());

    styleContainer.appendChild(block);

    if (styleImageCount % 2 === 1) {
	console.log('adding empty col');
	styleContainer.appendChild(document.querySelector('.empty-col').cloneNode());
    }
}

function uploadImage(selectionList, styleOrContent, styleImg=styleImage) {
    let choice = selectionList.value;
    fileUploader.styleOrContent = styleOrContent;
    if (choice === 'file-upload') {
	fileUploader.styleImage = styleImg;
	fileUploader.click();
    } else if (styleOrContent === 'content') {
	contentImage.src = '/assets/style-transfer/images/' + choice + '.jpg';
    } else {
	styleImg.src = '/assets/style-transfer/images/' + choice + '.jpg';
    }
}

//////////////////////////////////////////////////////////////////////
// Style Transfer Functions
function drawImageData(imageData) {
    let ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    canvas.width = imageData.width;
    canvas.height = Math.min(256, imageData.height);
    ctx.putImageData(imageData, 0, 0);
}

function getStyle(styleImages, styleWeights, contentStyleRatio) {
    let styles = styleImages.map(image => model.predictStyleParameters(image));
    let weightSum = styleWeights.reduce((a, b) => { return a + b; });
    styleWeights = styleWeights.map(weight => weight  / weightSum );
    let weightedStyles = styles.map((style, i) => style.mul(styleWeights[i]));
    let style = weightedStyles.reduce((a,b) => a.add(b));

    // Blend the style/content based on stylization strength prescribed by styleWeights
    let contentStyle = model.predictStyleParameters(contentImage).mul(contentStyleRatio);
    style = style.mul(1 - contentStyleRatio).add(contentStyle);

    return style
}

function getStyledImageData(contentImage, style) {
    let stylized = model.produceStylized(contentImage, style);
    return tf.browser.toPixels(stylized)
	.then((bytes) => new ImageData(bytes, stylized.shape[1], stylized.shape[0]));
}

async function stylize() {
    if (model.initialized === false) await model.initialize();
    console.log('Styling...');
    let styleImages = getStyleImages();
    let styleWeights = getStyleWeights();
    let contentStyleRatio = getContentStyleRatio(styleWeights);
    let style = getStyle(styleImages, styleWeights, contentStyleRatio);
    getStyledImageData(contentImage, style)
	.then(drawImageData)
    saveStylizedBtn.style.display = "block";
}
