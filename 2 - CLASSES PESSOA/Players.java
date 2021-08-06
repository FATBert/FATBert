package info.bytecraft.basiccommands;

import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.Table;

import org.bukkit.Bukkit;
import org.bukkit.entity.Player;

import com.avaje.ebean.validation.NotNull;

@Entity()
@Table(name="bytecraft_players")
public class Players {
	
	@Id
	private int id;
	
	@NotNull
	private String playerName;
	
	private String godColor;
	
	private boolean vanished;
	
	@NotNull
	private int kills;
	
	@NotNull
	private int deaths;
	
	@NotNull
	private String firstJoined;
	
	public int getId(){
		return this.id;
	}
	
	public void setId(int id){
		this.id = id;
	}
	
	public String getPlayerName(){
		return this.playerName;
	}
	
	public void setPlayerName(String name){
		this.playerName = name;
	}
	
	public Player getPlayer(){
		return Bukkit.getPlayer(playerName);
	}
	
	public void setPlayer(Player player){
		this.playerName = player.getName();
	}
	
	public String getGodColor(){
		return this.godColor;
	}
	
	public void setGodColor(String color){
		this.godColor = color;
	}
	
	public boolean isVanished(){
		return this.vanished;
	}
	
	public void setVanished(boolean vanished){
		this.vanished = vanished;
	}

	/**
	 * @return The amount of kills a player has.
	 */
	public int getKills() {
		return kills;
	}

	/**
	 * @param kills the kills to set
	 */
	public void setKills(int kills) {
		this.kills = kills;
	}

	public String getFirstJoined() {
		return firstJoined;
	}

	public void setFirstJoined(String firstJoined) {
		this.firstJoined = firstJoined;
	}

	/**
	 * @return The amount of times the player has died.
	 */
	public int getDeaths() {
		return deaths;
	}

	/**
	 * @param deaths the deaths to set
	 */
	public void setDeaths(int deaths) {
		this.deaths = deaths;
	}
}
