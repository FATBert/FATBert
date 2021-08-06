package lups;

import java.util.HashMap;

public class User {
	
	/*
	 * ringflags für die vers. Sicherheitsstufen
	 * 0 = Admin
	 * 1 = Prüfender
	 * 2 = Lehrender
	 * 3 = Student
	 */
	private int flag;
	
	private String passwort;
	
	private HashMap<String, String> userDaten = new HashMap<String, String>();
		
	private int semester = 0;
	
	private String fach = "";
	
	private HashMap<String, Double> leistunguebersicht = new HashMap<String, Double>();
		
	private HashMap<String, Boolean> aktuelleKurse = new HashMap<String, Boolean>();
	
	public User(HashMap<String, String> userDaten, String passwort, int flag) {
		this.userDaten.putAll(userDaten);
		this.passwort = passwort;
		this.flag = flag;
	}
	
	/**
	 * getter für user
	 */
	
	/**
	 * setPass(String) Setzt das Passwort.
	 * @param pass - Passwort
	 */
	public void setPass(String pass) {
		this.passwort = pass;
	}
	
	/**
	 * setUserDaten(HashMap<String,String>) Setzt Nutzerdaten.
	 * @param userDaten - Nutzerdaten(Name,Anschrift,Matnr,..)
	 */
	public void setUserDaten(HashMap<String, String> userDaten) {
		this.userDaten.putAll(userDaten);
	}
	
	/**
	 * setter nur für studenten; 
	 * initialisiert mit null
	 */
	
	/**
	 * setSemester(int) Setzt das Semester des Nutzers.
	 * @param semester
	 */
	public void setSemester(int semester) {
		this.semester = semester;
	}
	
	/**
	 * setFach(String) Setzt das Fach(Veranstaltung)
	 * @param fach
	 */
	public void setFach(String fach) {
		this.fach = fach;
	}
	
	/**
	 * setLeistung(HashMap<String, Double>) Setzt die Bewertung des Nutzers.
	 * @param leistung - Veranstaltung + Note
	 */
	public void setLeistung(HashMap<String, Double> leistung) {
		this.leistunguebersicht = leistung;
	}
	
	/**
	 * setKurse(HashMap<String, Boolean>) Setzt die Kurse.
	 * @param kurse - kursid und true, wenn zugelassen 
	 * 							 false, wenn angemeldet
	 */
	public void setKurse(HashMap<String, Boolean> kurse) {
		this.aktuelleKurse = kurse;
	}
	
	/**
	 * getter der user
	 */
	
	/**
	 * getFlag() Hohlt die Nutzergruppe
	 * @return flag
	 */
	public int getFlag() {
		return this.flag;
	}
	
//	public String getEmail() {
//		return this.email;
//	}
	
	/**
	 * getUserDaten() Hohlt die Nutzerdaten
	 * @return userData
	 */
	public HashMap<String, String> getUserDaten() {
		return this.userDaten;
	}
	
	/**
	 * getPass() Hohlt das aktuelle Passwort
	 * @return passwd
	 */
	public String getPass() {
		return this.passwort;
	}
	
	/**
	 * getter nur für studenten;
	 */
	
	/**
	 * getSemester() Hohlt das aktuelle Semester von Student
	 * @return semester
	 */
	public int getSemester() {
		return this.semester;
	}
	
	/**
	 * getFach() Hohlt die Veranstaltungen vom Studenten
	 * @return fach
	 */
	public String getFach() {
		return this.fach;
	}
	
	/**
	 * getLeistung() Hohlt die aktuellen Leistungen von Student
	 * @return leistung
	 */
	public HashMap<String, Double> getLeistung() {
		return this.leistunguebersicht;
	}
	
	/**
	 * getKurse() Hohlt die aktuellen Kurse bei denen Student angemeldet ist.
	 * @return kurse
	 */
	public HashMap<String, Boolean> getKurse() {
		return this.aktuelleKurse;
	}
}
