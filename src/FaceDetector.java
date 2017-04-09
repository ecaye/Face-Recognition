
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;

import javax.imageio.ImageIO;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;


public class FaceDetector { 
	
	private CascadeClassifier faceDetector;
	private boolean simsModeEnabled = false;
	private MatOfRect lastFacesDetected;
	private ArrayList<String> lastPeopleDetected = new ArrayList<>();
	
	public FaceDetector(){
        faceDetector = new CascadeClassifier("haarcascades/haarcascade_frontalface_alt.xml");
	}
	
    public void detectFaces(String dirPath, boolean outputOriginalImage, boolean renameAndMoveProcessedFile) {
 
        //System.out.println("\nRunning FaceDetector");
        
        // create CascadeClassifier with xml file for frontal face recognition
        
        File folder = new File(dirPath);
        File[] listOfFiles = folder.listFiles();
        
        int index = 0;
        int indexFace = 0;
                
        for (int i = 0; i < listOfFiles.length; i++) {
        	String fileName = listOfFiles[i].getName();
            if (listOfFiles[i].isFile()) {
            	
            	// timestamp to generate unique name file
            	String timestamp = new SimpleDateFormat("yyyyMMddhhmmss").format(new Date());
            	
                Mat originalImage = Imgcodecs.imread(dirPath + "/" + fileName);
                
                ArrayList<Mat> croppedFacesImages = detectAndCrop(originalImage);
                
                for (Mat croppedFaceImage : croppedFacesImages) {
                	// Resize photo
                	croppedFaceImage = resizeImage(croppedFaceImage, new Size(128,128));
                	
                	saveImage(timestamp + "_ouputFace" + (++indexFace) + ".png", croppedFaceImage);
        		}
                
                if(outputOriginalImage){
                	Mat FacesImage = detect(originalImage);
            		saveImage(timestamp + "ouputDetectedFaces" + (++index) + ".png", FacesImage);
                }
            	
            	//System.out.println("File " + listOfFiles[i].getName() + " processing \n");
            	
            	if(renameAndMoveProcessedFile){
            		listOfFiles[i].renameTo(new File("resources/processed/"+ timestamp +"_" + fileName));
            	}
            }
          }

    }
    
    /**
     * Save image in output folder
     * @param imageName
     * @param image
     */
    protected static void saveImage(String imageName, Mat image){
        //System.out.println(String.format("Writing %s", imageName));
        Imgcodecs.imwrite("resources/output/" + imageName, image);
    }
        
    /**
     * Detect only faces on original image (target them)
     * @param faceDetector
     * @param originalImage
     * @return originalImage with detected faces
     */
    public Mat detect(Mat originalImage){
        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(originalImage, faceDetections);
 
        //System.out.println(String.format("Detected %s faces", faceDetections.toArray().length));
        
        for (Rect rect : faceDetections.toArray()) {
        	// Draw rect
        	Imgproc.rectangle(originalImage, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));
        }
    	
		return originalImage;
    }
    
    
    /**
     * Detect faces on the original image and crop them (create one image per face)
     * @param faceDetector
     * @param originalImage
     * @return list of cropped faces
     */
    public ArrayList<Mat> detectAndCrop(Mat originalImage){
        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(originalImage, faceDetections);
 
        //System.out.println(String.format("Detected %s faces", faceDetections.toArray().length));
        
        ArrayList<Mat> croppedFacesImages = new ArrayList<Mat>();

        for (Rect rect : faceDetections.toArray()) {
        	croppedFacesImages.add(originalImage.submat(rect));
        }
    	
		return croppedFacesImages;
    }

	public Rect[] detectAndGuess(Mat originalImage, KNN knn) throws IOException {
		lastFacesDetected = new MatOfRect();
		faceDetector.detectMultiScale(originalImage, lastFacesDetected);
 
        //System.out.println(String.format("Detected %s faces", lastFacesDetected.toArray().length));
        
        MatOfByte mem = new MatOfByte();
        Mat face = new Mat();
        
        lastPeopleDetected.clear();
        
        for (Rect rect : lastFacesDetected.toArray()) {
        	
        	face = resizeImage(originalImage.submat(rect), new Size(128,128));
            Imgcodecs.imencode(".bmp", face, mem);
            
        	LittleImage testImage = new LittleImage();
            testImage.fromBmp((BufferedImage) ImageIO.read(new ByteArrayInputStream(mem.toArray())));
            
            String people = knn.guessPeople(testImage, 5);
            lastPeopleDetected.add(people);
        	
            if (simsModeEnabled) {
	        	displaySimsMode(originalImage, rect, people);
            }
            else {
	        	// Draw Rect
	        	Imgproc.rectangle(originalImage, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 0, 255));
	        	// Draw Text
	        	Imgproc.putText(originalImage, people, new Point(rect.x, rect.y), Core.FONT_HERSHEY_COMPLEX, 1.0, new Scalar(0,0,255));
            }
        }
    	
		return lastFacesDetected.toArray();
    }

    public Rect[] showLastFacesDetected(Mat originalImage) throws IOException {
		lastFacesDetected = new MatOfRect();
		faceDetector.detectMultiScale(originalImage, lastFacesDetected);
		
    	if (lastFacesDetected.toArray().length == lastPeopleDetected.size() && lastPeopleDetected.size() > 0) {
	        for (int i = 0; i < lastFacesDetected.toArray().length; ++i) {
	        	Rect rect = lastFacesDetected.toArray()[i];
	        	String people = lastPeopleDetected.get(i);
	        	
	            if (simsModeEnabled) {
		        	displaySimsMode(originalImage, rect, people);
	            }
	            else {
		        	// Draw Rect
		        	Imgproc.rectangle(originalImage, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 0, 255));
		        	// Draw Text
		        	Imgproc.putText(originalImage, people, new Point(rect.x, rect.y), Core.FONT_HERSHEY_COMPLEX, 1.0, new Scalar(0,0,255));
	            }
	        }
	    	
    	}
		return lastFacesDetected.toArray();
    }

	private void displaySimsMode(Mat originalImage, Rect rect, String people) {
		int markerSize = (int) (Math.ceil(rect.width / 4));
		if (markerSize > 600) {
			markerSize = 600;
		}
		
		Imgproc.drawMarker(originalImage, new Point(rect.x + rect.width / 2, rect.y - markerSize * 2.5), new Scalar(15,179,35), 3, markerSize, (int) (markerSize / 1.2), 1);
		Imgproc.drawMarker(originalImage, new Point(rect.x + rect.width / 2, rect.y - markerSize * 2.5), new Scalar(0,0,0), 3, markerSize * 2, 1, 1);
		Imgproc.putText(originalImage, people, new Point(rect.x + rect.width / 2 - 30, rect.y - markerSize * 4), Core.FONT_HERSHEY_SIMPLEX, 0.7, new Scalar(0, 0, 0));
	}

	public void toggleSimsMode() {
		simsModeEnabled = !simsModeEnabled;
	}
    
    /**
     * Resize image (in parameter)
     * @param image
     * @param size
     * @return resized image
     */
    protected static Mat resizeImage(Mat image, Size size){
        Mat resizeImage = new Mat();
        Imgproc.resize( image, resizeImage, size );
        return resizeImage;
    }
    
}
