
import java.awt.Graphics;
import java.awt.Image;
import java.awt.Toolkit;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.InetSocketAddress;
import java.net.Proxy;
import java.net.URL;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import javax.imageio.ImageIO;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.JSlider;
import javax.swing.event.ChangeEvent;
import javax.swing.event.ChangeListener;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;

public class FaceCam extends javax.swing.JFrame {

	private static final long serialVersionUID = 1L;

	private RealTimeThread realTimeThread = null;
	int count = 0;
	VideoCapture webSource = null;
	Mat frame = new Mat();
	FaceDetector faceDetector;

	private JButton recoButton;
	private JButton trainingButton;
	private JButton loadImageButton;
	private JButton loadFromUrlButton;
	private JButton loadFromFileButton;
	private javax.swing.JTextField textField;
	private javax.swing.JPanel jPanel1;
	private JCheckBox simsMode;

	private JSlider slider1;
	private JSlider slider2;
	private JSlider slider3;
	private JSlider slider4;
	private JSlider slider5;
	private JSlider slider6;
	// private int minH = 4, maxH = 8, minS = 35, maxS = 70, minV = 70, maxV =
	// 221;
	// private int minH = 0, maxH = 9, minS = 43, maxS = 255, minV = 70, maxV =
	// 221;
	//private int minH = 1, maxH = 133, minS = 60, maxS = 255, minV = 129, maxV = 149;
	private int minH = 0, maxH = 96, minS = 48, maxS = 255, minV = 127, maxV = 176;


	int inAngleMin = 200, inAngleMax = 300, angleMin = 180, angleMax = 359, lengthMin = 10, lengthMax = 80;

	private KNN knn;

	class RealTimeThread implements Runnable {

		protected volatile boolean runnable = false;
		protected volatile boolean isTrainingMode = false;
		protected volatile String label;
		protected volatile int indexFace;
		protected volatile long oldTime;

		public RealTimeThread(boolean trainingMode, String label) {
			isTrainingMode = trainingMode;
			if (label == null || label.isEmpty()) {
				this.label = "";
			} else {
				this.label = label + "/";
			}
		}

		private Rect[] detectFaceRealTime() throws IOException, InterruptedException {
			if (isTrainingMode) {
				ArrayList<Mat> croppedFaces = faceDetector.detectAndCrop(frame);
				if (croppedFaces.size() == 0)
					return null;
				String timestamp = new SimpleDateFormat("yyyyMMddhhmmss").format(new Date());
				String dirPath = "resources/trainingImages/" + label;
				String fileName = timestamp + "_ouputFace" + (++indexFace) + ".png";
				File folder = new File(dirPath);
				folder.mkdir();
				Imgcodecs.imwrite(dirPath + fileName,
						FaceDetector.resizeImage(croppedFaces.get(0), new Size(128, 128)));
			} else {
				Core.flip(frame, frame, 1);
				long currentTime = System.currentTimeMillis();

				Rect[] listOfFacesRect;

				if (currentTime - oldTime > 50 /* 1000 */) {
					oldTime = currentTime;
					listOfFacesRect = faceDetector.detectAndGuess(frame, knn);
				} else {
					listOfFacesRect = faceDetector.showLastFacesDetected(frame);
				}
				return listOfFacesRect;
			}
			return null;
		}

		private MatOfPoint convertIndexesToPoints(MatOfPoint contour, MatOfInt indexes) {
			int[] arrIndex = indexes.toArray();
			Point[] arrContour = contour.toArray();
			Point[] arrPoints = new Point[arrIndex.length];

			for (int i = 0; i < arrIndex.length; i++) {
				arrPoints[i] = arrContour[arrIndex[i]];
			}

			MatOfPoint hull = new MatOfPoint();
			hull.fromArray(arrPoints);
			return hull;
		}

		private double innerAngle(double x, double y, double x2, double y2, double x3, double y3) {

			double dist1 = Math.sqrt((x - x3) * (x - x3) + (y - y3) * (y - y3));
			double dist2 = Math.sqrt((x2 - x3) * (x2 - x3) + (y2 - y3) * (y2 - y3));

			double Ax, Ay;
			double Bx, By;
			double Cx, Cy;

			// find closest point to C
			// printf("dist = %lf %lf\n", dist1, dist2);

			Cx = x3;
			Cy = y3;
			if (dist1 < dist2) {
				Bx = x;
				By = y;
				Ax = x2;
				Ay = y2;

			} else {
				Bx = x2;
				By = y2;
				Ax = x;
				Ay = y;
			}

			double Q1 = Cx - Ax;
			double Q2 = Cy - Ay;
			double P1 = Bx - Ax;
			double P2 = By - Ay;

			double A = Math.acos((P1 * Q1 + P2 * Q2) / (Math.sqrt(P1 * P1 + P2 * P2) * Math.sqrt(Q1 * Q1 + Q2 * Q2)));

			A = A * 180 / Math.PI;

			return A;
		}

		public void detectHandRealTime(Rect[] listOfFacesRect) {

			Mat hsv = new Mat();
			Imgproc.cvtColor(frame, hsv, Imgproc.COLOR_BGR2HSV);

			for (Rect faceRect : listOfFacesRect) {
				Imgproc.rectangle(hsv, new Point(faceRect.x, faceRect.y),
						new Point(faceRect.x + faceRect.width, faceRect.y + faceRect.height), new Scalar(255, 255, 255),
						-1);
			}

			Core.inRange(hsv, new Scalar(minH, minS, minV), new Scalar(maxH, maxS, maxV), hsv);

			// Pre processing
			int blurSize = 5;
			int elementSize = 10;
			Imgproc.medianBlur(hsv, hsv, blurSize);
			Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, 
					new Size(2 * elementSize + 1, 2 * elementSize + 1), new Point(elementSize, elementSize));
			
			/*for(int nb = 0; nb < 5; nb++){
				Imgproc.dilate(hsv, hsv, element);
			}
			for(int nb = 0; nb < 5; nb++){
				Imgproc.erode(hsv, hsv, element);
			}*/
			
			 //frame = hsv; int tot = 0; if (tot == 0) return;

			// Contour detection
			List<MatOfPoint> contours = new ArrayList<>();
			MatOfInt4 hierarchy = new MatOfInt4();

			Imgproc.findContours(hsv, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE,
					new Point(0, 0));
			int largestContour = 0;
			for (int i = 1; i < contours.size(); i++) {
				if (Imgproc.contourArea(contours.get(i)) > Imgproc.contourArea(contours.get(largestContour)))
					largestContour = i;
			}
			Imgproc.drawContours(frame, contours, largestContour, new Scalar(0, 0, 255), 1);

			// Convex hull
			if (!contours.isEmpty()) {
				MatOfInt hull = new MatOfInt();
				Imgproc.convexHull(contours.get(largestContour), hull, false);

				MatOfPoint tempContour = contours.get(largestContour);

				int index = (int) hull.get(((int) hull.size().height) - 1, 0)[0];
				Point pt, pt0 = new Point(tempContour.get(index, 0)[0], tempContour.get(index, 0)[1]);
				for (int j = 0; j < hull.size().height - 1; j++) {
					index = (int) hull.get(j, 0)[0];
					pt = new Point(tempContour.get(index, 0)[0], tempContour.get(index, 0)[1]);
					Imgproc.line(frame, pt0, pt, new Scalar(255, 0, 100), 8);
					pt0 = pt;
				}

				if (hull.rows() > 2) {
					MatOfInt hullIndexes = new MatOfInt();

					Imgproc.convexHull(contours.get(largestContour), hullIndexes, true);
					MatOfInt4 convexityDefects = new MatOfInt4();
					Imgproc.convexityDefects(contours.get(largestContour), hullIndexes, convexityDefects);

					Rect boundingBox = Imgproc.boundingRect(convertIndexesToPoints(contours.get(largestContour), hull));
					Imgproc.rectangle(frame, new Point(boundingBox.x, boundingBox.y),
							new Point(boundingBox.x + boundingBox.width, boundingBox.y + boundingBox.height),
							new Scalar(255, 0, 0));
					Point center = new Point(boundingBox.x + boundingBox.width / 2,
							boundingBox.y + boundingBox.height / 2);
					List<Point> validPoints = new ArrayList<>();

					for (int i = 0; i < convexityDefects.rows(); i++) {
						double[] d1 = contours.get(largestContour).get((int) convexityDefects.get(i, 0)[0], 0);
						double[] d2 = contours.get(largestContour).get((int) convexityDefects.get(i, 0)[1], 0);
						double[] d3 = contours.get(largestContour).get((int) convexityDefects.get(i, 0)[2], 0);

						Point p1 = new Point(d1[0], d1[1]);
						Point p2 = new Point(d2[0], d2[1]);
						Point p3 = new Point(d3[0], d3[1]);

						Imgproc.line(frame, p1, p3, new Scalar(255, 0, 0), 2);
						Imgproc.line(frame, p3, p2, new Scalar(255, 0, 0), 2);

						double angle = Math.atan2(center.y - p1.y, center.x - p1.x) * 180 / Math.PI;
						double inAngle = innerAngle(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y);
						double length = Math.sqrt(Math.pow(p1.x - p3.x, 2) + Math.pow(p1.y - p3.y, 2));
						if (angle > angleMin - 180 && angle < angleMax - 180 && inAngle > inAngleMin - 180
								&& inAngle < inAngleMax - 180 && length > lengthMin / 100.0 * boundingBox.height
								&& length < lengthMax / 100.0 * boundingBox.height) {
							validPoints.add(p1);
						}
					}
					for (int i = 0; i < validPoints.size(); i++) {
						Imgproc.circle(frame, validPoints.get(i), 9, new Scalar(0, 255, 0), 2);
					}
				}
			}

		}

		@Override
		public void run() {
			synchronized (this) {

				indexFace = 0;
				oldTime = System.currentTimeMillis();

				while (runnable) {
					if (webSource.grab()) {
						try {
							webSource.retrieve(frame);
							Graphics g = jPanel1.getGraphics();

							Rect[] listOfFacesRect = detectFaceRealTime();

							detectHandRealTime(listOfFacesRect);

							BufferedImage buff = matToBufferedImage(frame);

							if (g.drawImage(buff, 0, 0, getWidth(), getHeight() - 150, 0, 0, buff.getWidth(),
									buff.getHeight(), null)) {
								if (runnable == false) {
									System.out.println("Paused ..... ");
									this.wait();
								}
							}

						} catch (IOException | InterruptedException e) {
							System.out.println("Error");
						}
					}
				}
			}
		}
	}

	/**
	 * Creates new form FaceDetection
	 */
	public FaceCam(KNN knn) {
		faceDetector = new FaceDetector();
		initComponents();
		this.knn = knn;
	}

	/**
	 * This method is called from within the constructor to initialize the form.
	 */
	private void initComponents() {

		// instantiates all graphic components
		jPanel1 = new javax.swing.JPanel();
		recoButton = new JButton();
		trainingButton = new JButton();
		textField = new javax.swing.JTextField(10);
		loadImageButton = new JButton();
		loadFromUrlButton = new JButton();
		loadFromFileButton = new JButton();
		simsMode = new JCheckBox();

		slider1 = new JSlider(JSlider.HORIZONTAL, 0, 255, minH);
		slider2 = new JSlider(JSlider.HORIZONTAL, 0, 255, maxH);
		slider3 = new JSlider(JSlider.HORIZONTAL, 0, 255, minS);
		slider4 = new JSlider(JSlider.HORIZONTAL, 0, 255, maxS);
		slider5 = new JSlider(JSlider.HORIZONTAL, 0, 255, minV);
		slider6 = new JSlider(JSlider.HORIZONTAL, 0, 255, maxV);

		slider1.setPaintLabels(true);
		slider1.setPaintTicks(true);
		slider1.setLabelTable(slider1.createStandardLabels(50));

		slider2.setPaintLabels(true);
		slider2.setPaintTicks(true);
		slider2.setLabelTable(slider1.createStandardLabels(50));

		slider3.setPaintLabels(true);
		slider3.setPaintTicks(true);
		slider3.setLabelTable(slider1.createStandardLabels(50));

		slider4.setPaintLabels(true);
		slider4.setPaintTicks(true);
		slider4.setLabelTable(slider1.createStandardLabels(50));

		slider5.setPaintLabels(true);
		slider5.setPaintTicks(true);
		slider5.setLabelTable(slider1.createStandardLabels(50));

		slider6.setPaintLabels(true);
		slider6.setPaintTicks(true);
		slider6.setLabelTable(slider1.createStandardLabels(50));

		slider1.addChangeListener(new ChangeListener() {
			@Override
			public void stateChanged(ChangeEvent e) {
				minH = slider1.getValue();
				System.out.println("minH : " + minH);
				System.out.println("maxH : " + maxH);
				System.out.println("minS : " + minS);
				System.out.println("maxS : " + maxS);
				System.out.println("minV : " + minV);
				System.out.println("maxV : " + maxV);
			}
		});
		slider2.addChangeListener(new ChangeListener() {
			@Override
			public void stateChanged(ChangeEvent e) {
				maxH = slider2.getValue();
				System.out.println("minH : " + minH);
				System.out.println("maxH : " + maxH);
				System.out.println("minS : " + minS);
				System.out.println("maxS : " + maxS);
				System.out.println("minV : " + minV);
				System.out.println("maxV : " + maxV);
			}
		});
		slider3.addChangeListener(new ChangeListener() {
			@Override
			public void stateChanged(ChangeEvent e) {
				minS = slider3.getValue();
				System.out.println("minH : " + minH);
				System.out.println("maxH : " + maxH);
				System.out.println("minS : " + minS);
				System.out.println("maxS : " + maxS);
				System.out.println("minV : " + minV);
				System.out.println("maxV : " + maxV);
			}
		});
		slider4.addChangeListener(new ChangeListener() {
			@Override
			public void stateChanged(ChangeEvent e) {
				maxS = slider4.getValue();
				System.out.println("minH : " + minH);
				System.out.println("maxH : " + maxH);
				System.out.println("minS : " + minS);
				System.out.println("maxS : " + maxS);
				System.out.println("minV : " + minV);
				System.out.println("maxV : " + maxV);
			}
		});
		slider5.addChangeListener(new ChangeListener() {
			@Override
			public void stateChanged(ChangeEvent e) {
				minV = slider5.getValue();
				System.out.println("minH : " + minH);
				System.out.println("maxH : " + maxH);
				System.out.println("minS : " + minS);
				System.out.println("maxS : " + maxS);
				System.out.println("minV : " + minV);
				System.out.println("maxV : " + maxV);
			}
		});
		slider6.addChangeListener(new ChangeListener() {
			@Override
			public void stateChanged(ChangeEvent e) {
				maxV = slider6.getValue();
				System.out.println("minH : " + minH);
				System.out.println("maxH : " + maxH);
				System.out.println("minS : " + minS);
				System.out.println("maxS : " + maxS);
				System.out.println("minV : " + minV);
				System.out.println("maxV : " + maxV);
			}
		});

		setExtendedState(JFrame.MAXIMIZED_BOTH);

		setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

		javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
		jPanel1.setLayout(jPanel1Layout);
		jPanel1Layout.setHorizontalGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
				.addGap(0, 0, Short.MAX_VALUE));
		jPanel1Layout.setVerticalGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
				.addGap(0, (int) (Toolkit.getDefaultToolkit().getScreenSize().getHeight() - 150), Short.MAX_VALUE));

		// ### buttons actions ###

		// Start&Stop recognition button
		recoButton.setText("Start Recognition");
		recoButton.addActionListener(new java.awt.event.ActionListener() {
			public void actionPerformed(java.awt.event.ActionEvent evt) {
				if (realTimeThread != null && realTimeThread.runnable) {
					trainingButton.setEnabled(true);
					recoButton.setText("Start Recognition");
					stopButtonActionPerformed(evt);
				} else {
					trainingButton.setEnabled(false);
					recoButton.setText("Stop Recognition");
					startButtonActionPerformed(evt, false, null);
				}
			}
		});

		// Start&Stop training button
		trainingButton.setText("Start Training");
		trainingButton.addActionListener(new java.awt.event.ActionListener() {
			public void actionPerformed(java.awt.event.ActionEvent evt) {
				if (realTimeThread != null && realTimeThread.runnable) {
					recoButton.setEnabled(true);
					loadImageButton.setEnabled(true);
					trainingButton.setText("Start Training");
					stopButtonActionPerformed(evt);
				} else {
					recoButton.setEnabled(false);
					trainingButton.setText("Stop Training");
					startButtonActionPerformed(evt, true, textField.getText());
				}
			}
		});

		// Load training images button
		loadImageButton.setText("Load Training Img");
		loadImageButton.addActionListener(new java.awt.event.ActionListener() {
			public void actionPerformed(java.awt.event.ActionEvent evt) {
				loadImageButtonActionPerformed(evt);
			}
		});

		// Load image from url button
		loadFromUrlButton.setText("Load From URL");
		loadFromUrlButton.addActionListener(new java.awt.event.ActionListener() {
			public void actionPerformed(java.awt.event.ActionEvent evt) {
				loadFromUrlButtonActionPerformed(evt);
			}
		});

		// Load image from file button
		loadFromFileButton.setText("Load From File");
		loadFromFileButton.addActionListener(new java.awt.event.ActionListener() {
			public void actionPerformed(java.awt.event.ActionEvent evt) {
				loadFromFileButtonActionPerformed(evt);
			}
		});

		// Sims mode (trick)
		simsMode.setText("Sims mode");
		simsMode.addActionListener(new java.awt.event.ActionListener() {
			public void actionPerformed(java.awt.event.ActionEvent evt) {
				faceDetector.toggleSimsMode();
			}
		});

		javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
		getContentPane().setLayout(layout);
		layout.setHorizontalGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
				.addGroup(layout.createSequentialGroup().addGap(24, 24, 24)
						.addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE,
								javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
						.addContainerGap())
				.addGroup(layout.createSequentialGroup().addComponent(slider1).addComponent(slider2)
						.addComponent(slider3).addComponent(slider4).addComponent(slider5).addComponent(slider6))
				.addGroup(layout.createSequentialGroup().addGap(255, 255, 255).addComponent(loadImageButton)
						.addGap(50, 50, 50).addComponent(recoButton).addGap(50, 50, 50).addComponent(textField)
						.addGap(5, 5, 5).addComponent(trainingButton).addGap(50, 50, 50).addComponent(loadFromUrlButton)
						.addGap(50, 50, 50).addComponent(loadFromFileButton).addGap(50, 50, 50).addComponent(simsMode)
						.addContainerGap(258, Short.MAX_VALUE)));
		layout.setVerticalGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
				.addGroup(layout.createSequentialGroup().addContainerGap()
						.addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE,
								javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
						.addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
						.addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
								.addComponent(slider1).addComponent(slider2).addComponent(slider3).addComponent(slider4)
								.addComponent(slider5).addComponent(slider6))
						.addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
								.addComponent(loadImageButton).addComponent(recoButton).addComponent(trainingButton)
								.addComponent(textField).addComponent(loadFromUrlButton)
								.addComponent(loadFromFileButton))
						.addComponent(simsMode)
						.addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)));

		pack();
	}

	private void stopButtonActionPerformed(java.awt.event.ActionEvent evt) {
		realTimeThread.runnable = false; // stop thread
		webSource.release(); // stop caturing fron cam
	}

	private void startButtonActionPerformed(java.awt.event.ActionEvent evt, boolean trainingMode, String str) {
		webSource = new VideoCapture(0); // video capture from default cam
		realTimeThread = new RealTimeThread(trainingMode, str);
		Thread t = new Thread(realTimeThread);
		t.setDaemon(true);
		realTimeThread.runnable = true;
		t.start(); // start thread
	}

	private void loadImageButtonActionPerformed(java.awt.event.ActionEvent evt) {
		trainingButton.setEnabled(false);
		recoButton.setEnabled(false);
		loadImageButton.setEnabled(false);

		knn.loadTrainingImages();

		trainingButton.setEnabled(true);
		recoButton.setEnabled(true);
	}

	private void loadFromUrlButtonActionPerformed(java.awt.event.ActionEvent evt) {
		Graphics g = jPanel1.getGraphics();

		String path = JOptionPane.showInputDialog("URL input");
		System.out.println("Get Image from " + path);
		if (path == null || path.isEmpty())
			return;

		try {

			// for PC from ESIEE (proxy)
			Proxy proxy = new Proxy(Proxy.Type.HTTP, new InetSocketAddress("147.215.1.189", 3128));
			HttpURLConnection connection = (HttpURLConnection) new URL(path).openConnection(proxy);

			BufferedImage buff = ImageIO.read(connection.getInputStream());

			Mat processedImage = bufferedImageToMat(buff);
			faceDetector.detectAndGuess(processedImage, knn);
			buff = matToBufferedImage(processedImage);

			g.clearRect(0, 0, getWidth(), getHeight());
			g.drawImage(buff, 0, 0, buff.getWidth(), buff.getHeight(), 0, 0, buff.getWidth(), buff.getHeight(), null);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private void loadFromFileButtonActionPerformed(java.awt.event.ActionEvent evt) {
		Graphics g = jPanel1.getGraphics();

		JFileChooser dialogue = new JFileChooser(new File("."));
		File fichier;
		String path;

		if (dialogue.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
			fichier = dialogue.getSelectedFile();
			path = fichier.getPath();
		} else {
			return;
		}

		System.out.println("Get Image from " + path);

		try {
			BufferedImage buff = ImageIO.read(fichier);
			Mat processedImage = bufferedImageToMat(buff);
			faceDetector.detectAndGuess(processedImage, knn);
			buff = matToBufferedImage(processedImage);

			g.clearRect(0, 0, getWidth(), getHeight());
			g.drawImage(buff, 0, 0, buff.getWidth(), buff.getHeight(), 0, 0, buff.getWidth(), buff.getHeight(), null);

		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	/**
	 * convert buffered image to Mat (OpenCv object)
	 * 
	 * @param buffered
	 *            image
	 * @return Mat
	 */
	public static Mat bufferedImageToMat(BufferedImage bi) {
		Mat mat = new Mat(bi.getHeight(), bi.getWidth(), CvType.CV_8UC3);
		byte[] data = ((DataBufferByte) bi.getRaster().getDataBuffer()).getData();
		mat.put(0, 0, data);
		return mat;
	}

	/**
	 * convert Mat (OpenCv object) to buffered image
	 * 
	 * @param Mat
	 * @return buffered image
	 */
	public static BufferedImage matToBufferedImage(Mat imageMat) throws IOException {
		MatOfByte mem = new MatOfByte();
		Imgcodecs.imencode(".bmp", imageMat, mem);
		Image im = ImageIO.read(new ByteArrayInputStream(mem.toArray()));
		return (BufferedImage) im;
	}

}
