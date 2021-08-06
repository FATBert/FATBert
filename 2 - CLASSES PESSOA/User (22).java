package model;

import java.io.Serializable;

/**
 * Class User
 * 
 * A class for holding users
 */

public class User implements Serializable {
	private int id;
	private String username;
	private String name;
	private String email;
	private String position;
	
	/**
	 * Full constructor
	 * 
	 * @param username
	 * @param id
	 * @param name
	 * @param email
	 * @param position		- The users position within the company 
	 */
	public User(int id, String username, String name, String email, String position) {
		setID(id);
		setUsername(username);
		setName(name);
		setEmail(email);
		setPosition(position);
	}
	
	/**
	 * Minimal constructor
	 * 
	 * @param username
	 * @param id
	 */
//	public User(int id, String username) {
//		setID(id);
//		setUsername(username);
//	}
	
	public User(int id, String name) {
	setID(id);
	setName(name);
	}
	
	
	
	public void setID(int id) {
		this.id = id;
	}
	
	public int getID() {
		return id;
	}
	
	public void setUsername(String username) {
		this.username = username;
	}
	
	public String getUsername() {
		return username;
	}
	
	public void setName(String name) {
		this.name = name;
	}
	
	public String getName() {
		return name;
	}
	
	public void setEmail(String email) {
		this.email = email;
	}
	
	public String getEmail() {
		return email;
	}
	
	public void setPosition(String position) {
		this.position = position;
	}
	
	public String getPosition() {
		return position;
	}
	
	public String toString(){
		return this.name;
	}
	
	@Override
	public boolean equals(Object obj) {
		if (obj instanceof User)
			return getName().equals(((User) obj).getName());
		else
			return false;
	}
	
	@Override
	public int hashCode() {
		return getName().hashCode();
	}
}