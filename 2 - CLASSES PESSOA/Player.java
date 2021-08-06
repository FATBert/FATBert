/**
 * 
 */
package de.hwr.damolog.model;

/**
 * @author nsmeenk
 * 
 */
public class Player {

	private String _name = "Spieler";
	private boolean _isAI = false;
	private int _winns = 0;
	private int _loses = 0;
	private int _remis = 0;

	/**
	 * Konstruktor
	 */
	public Player(String pName, int pWinns, int pLoses, int pRemis) {
		_name = pName;
		_winns = pWinns;
		_loses = pLoses;
		_remis = pRemis;
	}

	/**
	 * @return the name
	 */
	public String getName() {
		return _name;
	}

	/**
	 * @param pName
	 *            the name to set
	 */
	public void setName(String pName) {
		_name = pName;
	}

	/**
	 * @return the isAI
	 */
	public boolean isAI() {
		return _isAI;
	}

	/**
	 * @param pIsAI
	 *            the isAI to set
	 */
	public void setAI(boolean pIsAI) {
		_isAI = pIsAI;
	}

	/**
	 * @return the winns
	 */
	public int getWinns() {
		return _winns;
	}

	/**
	 * @return the loses
	 */
	public int getLoses() {
		return _loses;
	}

	/**
	 * @return the remis
	 */
	public int getRemis() {
		return _remis;
	}

	/**
	 * 
	 */
	public void addLose() {
		_loses++;
	}

	/**
	 * 
	 */
	public void addWinn() {
		_winns++;
	}

	/**
	 * 
	 */
	public void addRemi() {
		_remis++;
	}

	/**
	 * 
	 */
	public void resetStats() {
		_loses = 0;
		_remis = 0;
		_winns = 0;
	}

}
