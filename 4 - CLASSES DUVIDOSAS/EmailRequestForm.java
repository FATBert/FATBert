package com.llt.email.form;


public class EmailRequestForm {

	private String requestorEmaillAddress;
	private String domain;
	private String firstName;
	private String lastName;
	private String status;
	
	public String getRequestorEmaillAddress() {
		return requestorEmaillAddress;
	}
	public void setRequestorEmaillAddress(String requestorEmaillAddress) {
		this.requestorEmaillAddress = requestorEmaillAddress;
	}
	public String getDomain() {
		return domain;
	}
	public void setDomain(String domain) {
		this.domain = domain;
	}
	public String getFirstName() {
		return firstName;
	}
	public void setFirstName(String firstName) {
		this.firstName = firstName;
	}
	public String getLastName() {
		return lastName;
	}
	public void setLastName(String lastName) {
		this.lastName = lastName;
	}
	public String getStatus() {
		return status;
	}
	public void setStatus(String status) {
		this.status = status;
	}
	
}
