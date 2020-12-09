/*
variables
*/
var model;
var canvas;
var coords = [];
var mousePressed = false;
var mode;



/*
prepare the drawing canvas
*/
$(function() {
    canvas = window._canvas = new fabric.Canvas('canvas');
    canvas.backgroundColor = '#ffffff';
    canvas.isDrawingMode = 0;
    canvas.freeDrawingBrush.color = "black";
    canvas.freeDrawingBrush.width = 2;
    canvas.renderAll();
    //setup listeners
    canvas.on('mouse:up', function(e) {
        getFrame();
        mousePressed = false
    });
    canvas.on('mouse:down', function(e) {
        mousePressed = true
    });
    canvas.on('mouse:move', function(e) {
        recordCoor(e)
    });
})


/*
record the current drawing coordinates
*/
function recordCoor(event) {
    var pointer = canvas.getPointer(event.e);
    var posX = pointer.x;
    var posY = pointer.y;

    if (posX >= 0 && posY >= 0 && mousePressed) {
        coords.push(pointer)
    }
}

/*
get the best bounding box by trimming around the drawing
*/
//function getMinBox() {
//    //get coordinates
//    var coorX = coords.map(function(p) {
//        return p.x
//    });
//    var coorY = coords.map(function(p) {
//        return p.y
//    });
//
//    //find top left and bottom right corners
//    var min_coords = {
//        x: Math.min.apply(null, coorX),
//        y: Math.min.apply(null, coorY)
//    }
//    var max_coords = {
//        x: Math.max.apply(null, coorX),
//        y: Math.max.apply(null, coorY)
//    }
//
//    //return as struct
//    return {
//        min: min_coords,
//        max: max_coords
//    }
//}

/*
get the current image data
//*/
//function getImageData() {
//        //get the minimum bounding box around the drawing
//        const mbb = getMinBox()
//
//        //get image data according to dpi
//        const dpi = window.devicePixelRatio
//        const imgData = canvas.contextContainer.getImageData(mbb.min.x * dpi, mbb.min.y * dpi,
//                                                      (mbb.max.x - mbb.min.x) * dpi, (mbb.max.y - mbb.min.y) * dpi);
//        return imgData
//    }


/*
load the model
*/
async function start(cur_mode) {
    mode = cur_mode
    //load the model
    model = await tf.loadLayersModel('static/model2/model.json')
    allowDrawing()
}

/*
allow drawing on canvas
*/
function allowDrawing() {
    canvas.isDrawingMode = 1;
    if (mode == 'en')
        document.getElementById('status').innerHTML = 'Model Loaded';
    else
        document.getElementById('status').innerHTML = 'تم التحميل';
    $('button').prop('disabled', false);
    var slider = document.getElementById('myRange');
    slider.oninput = function() {
        canvas.freeDrawingBrush.width = this.value;
    };
}

/*
clear the canvs
*/
function erase() {
//    window.alert ("High");
    canvas.clear();
    canvas.backgroundColor = '#ffffff';
    coords = [];
}

function download_image_display(){

//	alert ("High");
	var canvas = document.getElementById("canvas");
	image = canvas.toDataURL("image/png").replace("image/png", "image/octet-stream");
	var httpRequest = new XMLHttpRequest();
	httpRequest.open("POST", "/imageupload");
	httpRequest.send(image);

    httpRequest.onreadystatechange = function() { // Call a function when the state changes.
    if (this.readyState === XMLHttpRequest.DONE && this.status === 200) {
        // Request finished. Do processing here.

        var response = this.response
        console.log(response);
        document.getElementById('responseId').innerHTML=response;
    }
}
}

