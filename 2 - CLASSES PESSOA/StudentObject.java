/**
 * Represents a student. Holds the students name,
 * student ID, grade, phone number, address, school,
 * site, program, and emergency contact information.
 */

package edu.upenn.cis350;

public class StudentObject {

	private String studentid;
	private int gradeLevel;
	private String name;
	private String phone;
	private String address;
	private String school;
	private String site;
	private String program;
	private String contactName;
	private String contactType;
	private String attendance_status;
	private String comments;
	
	public StudentObject(){
		
	}
	public StudentObject(String studentid, int gradeLevel, String name,
			String phone, String address, String school, String site,
			String program, String contactName, String contactType) {
		this.studentid = studentid;
		this.gradeLevel = gradeLevel;
		this.name = name;
		this.phone = phone;
		this.address = address;
		this.school = school;
		this.site = site;
		this.program = program;
		this.contactName = contactName;
		this.contactType = contactType;
		this.attendance_status = "Absent";
		this.setComments("");
	}
	
	public StudentObject(String studentid, String name){
		this.studentid = studentid;
		this.name = name;
	}
	
	public StudentObject(String name){
		this.name = name;
	}

	public String getName() {
		return name;
	}

	public void setName(String name) {
		this.name = name;
	}

	public String getPhone() {
		return phone;
	}

	public void setPhone(String phone) {
		this.phone = phone;
	}

	public String getAddress() {
		return address;
	}

	public void setAddress(String address) {
		this.address = address;
	}

	public String getStudentid() {
		return studentid;
	}

	public void setStudentid(String studentid) {
		this.studentid = studentid;
	}

	public String getSite() {
		return site;
	}

	public void setSite(String site) {
		this.site = site;
	}

	public int getGradeLevel() {
		return gradeLevel;
	}

	public void setGradeLevel(int gradeLevel) {
		this.gradeLevel = gradeLevel;
	}

	public String getSchool() {
		return school;
	}

	public void setSchool(String school) {
		this.school = school;
	}

	public String getProgram() {
		return program;
	}

	public void setProgram(String program) {
		this.program = program;
	}

	public String getContactName() {
		return contactName;
	}

	public void setContactName(String contactName) {
		this.contactName = contactName;
	}

	public String getContactType() {
		return contactType;
	}

	public void setContactType(String contactType) {
		this.contactType = contactType;
	}
	public String getStatus() {
		return attendance_status;
	}
	public void setStatus(String status) {
		this.attendance_status = status;
	}
	public String getComments() {
		return comments;
	}
	public void setComments(String comments) {
		this.comments = comments;
	}
}
