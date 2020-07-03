/* QuPath-Script to export annotations to label tif images (e.g. to be used for stardist) 

 Use "Run for project" to export annotations for all images within a QuPath project

 Afterwards both images and mask tiffs will be stored in the project subfolder 
 
 ground_truth
 ├── images
 └── masks
 
 Based on code by Olivier Burri, Romain Guiet (both https://biop.epfl.ch/) and https://forum.image.sc/t/export-qupath-annotations-for-stardist-training/37391/3
 
*/


// USER SETTINGS
def channel_of_interest = 1 // null to export all the channels 
def downsample = 1


def image_name = getProjectEntry().getImageName()


def rm = RoiManager.getRoiManager() ?: new RoiManager()
// create an annotation of the entire image
createSelectAllObject(true)
def fullimage_annotation = getSelectedObject()


imageData = getCurrentImageData();
server = imageData.getServer();
viewer = getCurrentViewer();
hierarchy = getCurrentHierarchy();

request = RegionRequest.createInstance(imageData.getServerPath(), downsample, fullimage_annotation.getROI())
pathImage = null;
pathImage = IJExtension.extractROIWithOverlay(server, fullimage_annotation, hierarchy, request, false, viewer.getOverlayOptions());
image = pathImage.getImage()

// Create the Labels image
def labels = IJ.createImage( "Labels", "16-bit black", image.getWidth(), image.getHeight() ,1 );
        
IJ.run(image, "To ROI Manager", "")
def rois = rm.getRoisAsArray() as List

def label_ip = labels.getProcessor()
def idx = 0
rois.each{ roi ->
   if (roi.getType() == Roi.RECTANGLE) {
       println("Ignoring Rectangle")
   } else {
    label_ip.setColor( ++idx )
    label_ip.setRoi( roi )
    label_ip.fill( roi )
   }
}
labels.setProcessor( label_ip )
        
 // Split to keep only channel of interest
def output = image
if  ( channel_of_interest != null){
    imp_chs =  ChannelSplitter.split( image )
    output = imp_chs[  channel_of_interest - 1 ]
}

saveImages(output, labels, image_name)
                
println( image_name + " Image and Mask Saved." )
        
// Save some RAM
output.close()
labels.close()
image.close()

// remove the fullimage_annotation from the list of annotations
// (true is to keep descendant objects)
removeObject(  fullimage_annotation,true )



// This will save the images in the selected folder
def saveImages(def images, def labels, def name) {
    def source_folder = new File ( buildFilePath( PROJECT_BASE_DIR, 'ground_truth', 'images' ) )
    def target_folder = new File ( buildFilePath( PROJECT_BASE_DIR, 'ground_truth', 'masks' ) )
    mkdirs( source_folder.getAbsolutePath() )
    mkdirs( target_folder.getAbsolutePath() )
    
    IJ.save( images , new File ( source_folder, name ).getAbsolutePath()+'.tif' )
    IJ.save( labels , new File ( target_folder, name ).getAbsolutePath()+'.tif' )

}


// Manage Imports
import qupath.lib.roi.RectangleROI
import qupath.imagej.gui.IJExtension;
import ij.IJ
import ij.gui.Roi
import ij.plugin.ChannelSplitter
import ij.plugin.frame.RoiManager
print "done"