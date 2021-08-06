package be.uclouvain.yasmim.entity;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.JoinColumn;
import javax.persistence.JoinTable;
import javax.persistence.ManyToMany;
import javax.persistence.Table;
import javax.persistence.Temporal;
import javax.persistence.Transient;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;

/**
 * Represents the user of the application.
 * @author Hildeberto Mendonca
 */
@Entity
@Table(name="user_account")
@XmlType(name="UserAccountType")
@XmlAccessorType(XmlAccessType.NONE)
public class UserAccount implements Serializable {
    private static final long serialVersionUID = 1L;

    @XmlAttribute(name="user_id")
    private String userId;
    @XmlElement
    private String firstName;
    @XmlElement
    private String lastName;
    private Integer sex;
    private Date birthDate;
    private String password;
    private String confirmationCode;
    private List<UserGroup> userGroups;

    public UserAccount() {}

    public UserAccount(String userId) {
        this.userId = userId;
    }

    @Id
    @Column(name="user_id")
    public String getUserId() {
        return this.userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    @Column(nullable=false)
    public String getPassword() {
        return password;
    }

    public void setPassword(String password) {
        this.password = password;
    }

    @Column(name="first_name", nullable=false)
    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    @Column(name="last_name", nullable=false)
    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    @Column(nullable=false)
    public Integer getSex() {
        return sex;
    }

    public void setSex(Integer sex) {
        this.sex = sex;
    }

    @Temporal(javax.persistence.TemporalType.DATE)
    @Column(name="birth_date",nullable=false)
    public Date getBirthDate() {
        return birthDate;
    }

    public void setBirthDate(Date birthDate) {
        this.birthDate = birthDate;
    }

    @Transient
    public int getAge() {
        if(this.birthDate != null) {
            Date today = Calendar.getInstance().getTime();
            return (int)(((((today.getTime() - birthDate.getTime()) / 1000) / 60) / 60) / 24) / 365;
        }
        return 0;
    }

    @Transient
    public String getEmailAddress() {
        return userId;
    }

    @ManyToMany
    @JoinTable(
        name="user_group",
        joinColumns=@JoinColumn(name="user_id"),
        inverseJoinColumns=@JoinColumn(name="group_id"))
    public List<UserGroup> getUserGroups() {
        return userGroups;
    }

    public void setUserGroups(List<UserGroup> userGroups) {
        this.userGroups = userGroups;
    }

    public void addUserGroup(UserGroup userGroup) {
        if(this.userGroups == null)
            this.userGroups = new ArrayList<UserGroup>();
        this.userGroups.add(userGroup);
    }

    @Override
    public String toString() {
        return firstName + " "+ lastName;
    }

    @Column(name="confirmation_code")
    public String getConfirmationCode() {
        return confirmationCode;
    }

    public void setConfirmationCode(String confirmationCode) {
        this.confirmationCode = confirmationCode;
    }
}