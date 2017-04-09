
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;

import javax.imageio.ImageIO;

public class KNN {
	
	// Constantes
	private static final int LITTLE_IMAGE_SIZE = 128;
	private static final int NB_RGB_DIMENSIONS = 3;
	// Images folder
	private static final String TRAINING_IMAGE_FOLDER = "resources/trainingImages";
	
	private LittleImage[] trainingImages;
	public void loadTrainingImages(){
		long startTime;
		
		trainingImages = new LittleImage[getFilesCount(TRAINING_IMAGE_FOLDER)];

		// On charge les images
		startTime = System.currentTimeMillis();
		try {
			loadImages(trainingImages, TRAINING_IMAGE_FOLDER);
		} catch (IOException e) {
			e.printStackTrace();
		}
		calculateAndDisplayTimeElapsed(startTime, "Images chargées");
	}
	
	public static int getFilesCount(String dirPath) {
		File folder = new File(dirPath);
		File[] files = folder.listFiles();
		int count = 0;
		for (File f : files){
			
			if (f.getName().startsWith(".")) continue; // to prevent hidden file on mac !
			
			if (f.isDirectory() && !f.getName().startsWith("."))
				count += getFilesCount(f.getPath());
			else
				count++;
		}
		return count;
	}

	
	public String guessPeople(LittleImage testImage, int k) {
		//System.out.println("k-NN avec k=" + k + " commencé.");
		
		if (trainingImages == null) {
			loadTrainingImages();
		}

		if(trainingImages.length == 0) return "";
		
		int[] pixelsTestImage = testImage.getPixels();

		int distanceIndex = 0;
		Object[][] distances = new Object[trainingImages.length][2];

		for (LittleImage trainingImage : trainingImages) {
			int[] pixelsTrainingImage = trainingImage.getPixels();

			int distance = 0;
			for (int i = 0; i < pixelsTestImage.length; i++) {
				distance += Math.abs(pixelsTestImage[i] - pixelsTrainingImage[i]);
			}

			// on a la distance
			distances[distanceIndex][0] = distance;
			distances[distanceIndex][1] = trainingImage.getLabel();
			distanceIndex++;
		}

		Arrays.sort(distances, new Comparator<Object[]>() {
			public int compare(Object[] obj1, Object[] obj2) {
				Integer numOfKeys1 = (Integer) obj1[0];
				Integer numOfKeys2 = (Integer) obj2[0];
				return numOfKeys1.compareTo(numOfKeys2);
			}
		});

		String[] labels = new String[k];
		for (int i = 0; i < k; i++) {
			labels[i] = ((Label) distances[i][1]).toString();
		}
		String predictedLabel = findPopularString(labels);
		
		//System.out.println(testImage.getLabel().name() + " -> " + predictedLabel);	
		return predictedLabel;
	}
	
	/**
	 * Retourne la string la plus présente dans un tableau de strings.
	 * 
	 * @param stringArray
	 *            le tableau de strings
	 * @return la string la plus présente
	 */
	public static String findPopularString(String[] stringArray) {
		if (stringArray == null || stringArray.length == 0)
			return null;
		Arrays.sort(stringArray);
		String previous = stringArray[0];
		String popular = stringArray[0];
		int count = 1;
		int maxCount = 1;
		for (int i = 1; i < stringArray.length; i++) {
			if (stringArray[i].equals(previous))
				count++;
			else {
				if (count > maxCount) {
					popular = stringArray[i - 1];
					maxCount = count;
				}
				previous = stringArray[i];
				count = 1;
			}
		}
		return count > maxCount ? stringArray[stringArray.length - 1] : popular;
	}

	
	/**
	 * Charge les images.
	 * 
	 * @param trainingImages
	 * @param folderName
	 * @throws IOException
	 */
	private static void loadImages(LittleImage[] trainingImages, String dirPath) throws IOException {
		System.out.println("Chargement des images. ( " + dirPath + " )");

		// On spécifie le nom du dossier ou trouver les images
		File resFolder = new File(dirPath);

		int trainingImageIndex = 0;
		for (File file : resFolder.listFiles()) {
			if(file.isDirectory()){
				// On récupère le nom de la classe
				String className = file.getName();
				System.out.println("-> images de la classe " + className);
				
				File folder = new File(dirPath + "/" + className);
				if(className.startsWith("."));
				
				for (File imageFile : folder.listFiles()) {
					if(imageFile.getName().startsWith(".")) continue;
					
					BufferedImage bufferedImage = ImageIO.read(imageFile);
					int imageWidth = bufferedImage.getWidth();
					int imageHeight = bufferedImage.getHeight();

					int[] pixels = new int[NB_RGB_DIMENSIONS];
					int pixelIndex = 0;
					
					int[] image = new int[LITTLE_IMAGE_SIZE * LITTLE_IMAGE_SIZE * NB_RGB_DIMENSIONS];

					for (int i = 0; i < imageWidth; i++) {
						for (int j = 0; j < imageHeight; j++) {
							int pixel = bufferedImage.getRGB(i, j);
							pixels = pixelToRgbArray(pixel);
							
							image[pixelIndex++] = pixels[0];
							image[pixelIndex++] = pixels[1];
							image[pixelIndex++] = pixels[2];	
						}
					}	
					trainingImages[trainingImageIndex++] = new LittleImage(image, Label.fromVal(className)); 
				}
			}
		}
		System.out.println(trainingImageIndex + " images chargées. \n");
	}

	/**
	 * Transforme la valeur d'un pixel en tableau RGB
	 * 
	 * @param pixel
	 *            valeur du pixel. Ex: -18726231621
	 * @return tableau RGB. Ex: [255, 100, 87]
	 */
	public static int[] pixelToRgbArray(int pixel) {
		int red = (pixel >> 16) & 0xff;
		int green = (pixel >> 8) & 0xff;
		int blue = (pixel) & 0xff;
		int[] RGB = { red, green, blue };
		return RGB;
	}

	/**
	 * Calcule et affiche le temps écoulé depuis un temps de départ.
	 * 
	 * @param startTime
	 *            le temps au départ
	 */
	private static void calculateAndDisplayTimeElapsed(long startTime, String message) {
		long endTime = System.currentTimeMillis();

		// Temps en secondes
		int totalRunningTime = (int) (((float) (endTime - startTime)) / 1000f);
		String unite = "secondes";

		// Temps en minutes si plus d'une minute
		if (totalRunningTime > 59) {
			totalRunningTime = (int) (totalRunningTime / 60);
			unite = "minutes";
		}

		System.out.println("=================");
		System.out.println(message + " en " + totalRunningTime + " " + unite + ".");
		System.out.println("=================\n\n");
	}

	/**
	 * Calcule et affiche le temps d'éxecution estimé.
	 * 
	 * @param startTime
	 *            le temps au départ
	 */
	private static void estimateAndDisplayTimeNeeded(long startTime) {
		long endTime = System.currentTimeMillis();

		// Temps en secondes
		int totalRunningTime = (int) (((float) (endTime - startTime)) / 1000f);
		totalRunningTime = totalRunningTime * 100;
		String unite = "secondes";

		// Temps en minutes si plus d'une minute
		if (totalRunningTime > 59) {
			totalRunningTime = (int) (totalRunningTime / 60);
			unite = "minutes";
		}

		System.out.println("=================");
		System.out.println("-> Le programme devrait se terminer en " + totalRunningTime + " " + unite + ".");
		System.out.println("=================");
	}
}
