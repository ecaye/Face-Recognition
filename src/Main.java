
import java.io.IOException;

import org.opencv.core.Core;

public class Main {

	public static void main(String[] args) throws IOException {
		System.out.println(System.getProperty("java.library.path"));

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		final KNN knn = new KNN();		
        
        try {
        	// Use "Nimbus" for IHM
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (Exception ex) {
            java.util.logging.Logger.getLogger(FaceCam.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }

        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new FaceCam(knn).setVisible(true);
            }
        });
        
	}

}
