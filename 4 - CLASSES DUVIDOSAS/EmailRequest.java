package com.llt.email.model;

import java.sql.Timestamp;

import org.hibernate.validator.constraints.Email;
import org.hibernate.validator.constraints.NotEmpty;

import com.llt.email.annotation.Column;

public class EmailRequest {
	
	@Column(name="emailRequestId")
	private Integer emailRequestId;
	
	@Column(name="requestorEmailAddress")
	@Email
	@NotEmpty (message="Please enter a valid Requester Email Address.")
	private String requestorEmailAddress;
	
	@Column(name="domain")
	@NotEmpty (message="Please enter a valid Domain.")
	private String domain;
	
	@Column(name="firstName")
	@NotEmpty (message="Please enter a valid First Name.")		
	private String firstName;
	
	@Column(name="lastName")
	@NotEmpty (message="Please enter a valid Last Name.")	
	private String lastName;
	
	@Column(name="status")
	private String status;
	
	@Column(name="createdDate")
	private Timestamp createdDate;
	
	@Column(name="updatedDate")
	private Timestamp updatedDate;
	@Column(name="validEmailAddress")
	private String validEmailAddress;
	
	public Integer getEmailRequestId() {
		return emailRequestId;
	}
	public void setEmailRequestId(Integer emailRequestId) {
		this.emailRequestId = emailRequestId;
	}
	public String getRequestorEmailAddress() {
		return requestorEmailAddress;
	}
	public void setRequestorEmailAddress(String requestorEmailAddress) {
		this.requestorEmailAddress = requestorEmailAddress;
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
	public Timestamp getCreatedDate() {
		return createdDate;
	}
	public void setCreatedDate(Timestamp createdDate) {
		this.createdDate = createdDate;
	}
	public Timestamp getUpdatedDate() {
		return updatedDate;
	}
	public void setUpdatedDate(Timestamp updatedDate) {
		this.updatedDate = updatedDate;
	}
	
	@Override
	public String toString() {
		StringBuffer string = new StringBuffer();
		
		return string.append("Domain = " + domain +"\r\n"
				+"FirstName = " + firstName + "\r\n" 
				+"LastName = " + lastName + "\r\n"
				+"RequestorEmailAddress = " + requestorEmailAddress + "\r\n"
				+"Status = " + status).toString();
	}

	public String getValidEmailAddress() {
		return validEmailAddress;
	}
	public void setValidEmailAddress(String validEmailAddress) {
		this.validEmailAddress = validEmailAddress;
	}
}