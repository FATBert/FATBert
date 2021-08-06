/**************************************************************************
 * This file is part of MCbb.                                              
 * MCbb is free software: you can redistribute it and/or modify            
 * it under the terms of the GNU General Public License as published by    
 * the Free Software Foundation, either version 3 of the License, or       
 * (at your option) any later version.                                     
 * MCbb is distributed in the hope that it will be useful,                 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of          
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the           
 * GNU General Public License for more details.                            
 * You should have received a copy of the GNU General Public License       
 * along with MCbb.  If not, see <http://www.gnu.org/licenses/>.           
 *************************************************************************/

package de.javakara.manf.api;

public class User {
	private int id,type,gid;
	private String name;
	
	public User(String name) {
		this.name = name;
	}

	public int getId() {
		return id;
	}
	
	public String getName() {
		return name;
	}

	public int getGroupId() {
		return gid;
	}
	
	public int getTypeId() {
		return type;
	}
	
	public void setGroupId(int gid) {
		this.gid = gid;
	}

	public void setUserType(int type) {
		this.type = type;
	}

	public void setUserId(int id) {
		this.id = id;
	}
}
