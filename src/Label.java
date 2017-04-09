
public enum Label {

	RAPH,
	SHABIR,
	MARCO,
	SENTHU,
	ERWIN;
		
	public static Label fromVal(String className) {
		return Label.valueOf(className.toUpperCase());
	}
}