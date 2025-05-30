/* QuPath-Script to export annotations for multiclass segmentation using StarDist
This is a modification of https://raw.githubusercontent.com/stardist/stardist/master/extras/qupath_export_annotations.groovy
but modified for multiclass scenario.
Use "Run for project" to export annotations for all images within a QuPath project.

Result:
 gt
 ├── images  <- Original images in .tif format
 ├── masks   <- 16 bit masks in .tif format
 └── dicts   <- Jsons containing dictionaries class of each mask instance

Json format:
{"1": "Class1",
"2": "Class2",
"3": "Class1",
...
}

Importing dicts using python:
```python=
path2json = "..."
def keystoint(x):
    return {int(k): v for k, v in x.items()}
with open(path2json, "r") as file:
    result = json.load(file, object_hook=keystoint)
```
Importing masks using python as uint16:
```python=
mask_path = "..."
mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
```
*/

double downsample = 1

def imageData = getCurrentImageData()
def server = imageData.getServer()
def metadata = server.getMetadata()
viewer = getCurrentViewer()
hierarchy = getCurrentHierarchy()
img_widht = metadata['width']
img_height = metadata['height']

def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())

createSelectAllObject(true)
def fullimage_annotation = getSelectedObject()
request = RegionRequest.createInstance(imageData.getServerPath(), downsample, fullimage_annotation.getROI())
pathImage = null
pathImage = IJExtension.extractROIWithOverlay(server, fullimage_annotation, hierarchy, request, false, viewer.getOverlayOptions())
image = pathImage.getImage()
def labels = IJ.createImage('Labels', '16-bit black', image.getWidth(), image.getHeight(), 1)
def label_ip = labels.getProcessor()

def annotationMap = [:]
def idx = 0
for (annotation in getAnnotationObjects()) {
    class_name = annotation.getPathClass()
    if (class_name == null) {
        println('Ignoring Rectangle or other shapes without class')
    } else {
        roi = annotation.getROI()
        def inputPolygonRoi
        roi.getVertices().each { vertex ->
            List<Float> xList = vertex.getX()
            List<Float> yList = vertex.getY()

            // Convert float lists to integer arrays
            int[] xArray = xList.collect { it.toInteger() } as int[]
            int[] yArray = yList.collect { it.toInteger() } as int[]

            inputPolygonRoi = new PolygonRoi(xArray, yArray, xArray.size(), Roi.POLYGON)
        }
        idx_n = ++idx
        annotationMap[idx_n] = class_name['name']
        label_ip.setColor(idx_n)
        label_ip.setRoi(inputPolygonRoi)
        label_ip.fill(inputPolygonRoi)
    }
}

saveResults(image, labels, name, annotationMap)

println('Image, mask and class dict are saved.')

// Save some RAM
image.close()
labels.close()

// (true is to keep descendant objects)
removeObject(fullimage_annotation, true)

def saveResults(def images, def labels, def name, def ann_map) {
    def source_folder = new File(buildFilePath(PROJECT_BASE_DIR, 'gt', 'images'))
    def target_folder = new File(buildFilePath(PROJECT_BASE_DIR, 'gt', 'masks'))
    def dict_folder = new File(buildFilePath(PROJECT_BASE_DIR, 'gt', 'dicts'))
    mkdirs(source_folder.getAbsolutePath())
    mkdirs(target_folder.getAbsolutePath())
    mkdirs(dict_folder.getAbsolutePath())

    IJ.save(images , new File(source_folder, name).getAbsolutePath() + '.tif')
    IJ.save(labels , new File(target_folder, name).getAbsolutePath() + '.tif')

    def jsonOutput = GsonTools.getInstance(true).toJson(ann_map)

    def outputFile = new File(dict_folder, name).getAbsolutePath() + '.json'
    new File(outputFile).text = jsonOutput

    println("JSON data saved to: ${outputFile}")
}

// Manage Imports
import qupath.imagej.gui.IJExtension
import ij.IJ
import ij.gui.Roi
import ij.gui.PolygonRoi

print 'done'
