//*******************************************************************
// Date: July-2021
// Credits: StarDist, DeepImageJ
// URL: 
//      https://github.com/stardist/stardist
//      https://deepimagej.github.io/deepimagej
// This macro was adapted from
// https://github.com/deepimagej/imagej-macros/blob/master/StarDist2D_Post-processing.ijm
// Please cite the respective contributions when using this code.
//*******************************************************************
//  Macro to run a the StarDist postprocessing on 2D images. 
//  StarDist and deepImageJ plugins need to be installed.
//  The macro assumes that the image to process is a stack in which 
//  the first channel corresponds to the object detection probability 
//  map and the remaining channels are the oriented distances from the
//  object border to its center.
//*******************************************************************

// Get the name of the image to call it
getDimensions(width, height, channels, slices, frames);
name=getTitle();

// these values will be replaced
probThresh=0.5;
nmsThresh=0.5;

// Isolate the detection probability scores
run("Make Substack...", "channels=1");
rename("scores");

// Isolate the oriented distances
run("Fire");
selectWindow(name);
run("Delete Slice", "delete=channel");
selectWindow(name);
run("Properties...", "channels=" + maxOf(channels, slices) - 1 + " slices=1 frames=1 pixel_width=1.0000 pixel_height=1.0000 voxel_depth=1.0000");
rename("distances");
run("royal");

// Run StarDist plugin
run("Command From Macro", "command=[de.csbdresden.stardist.StarDist2DNMS], args=['prob':'scores', 'dist':'distances', 'probThresh':'" + probThresh + "', 'nmsThresh':'" + nmsThresh + "', 'outputType':'Both', 'excludeBoundary':'2', 'roiPosition':'Stack', 'verbose':'false'], process=[false]");
