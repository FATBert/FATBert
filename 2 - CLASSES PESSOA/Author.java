package be.uclouvain.yasmim.entity;

import java.io.Serializable;
import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.Table;
import javax.persistence.Transient;
import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlAttribute;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlType;

@Entity
@Table(name="author")
@XmlType(name="AuthorType")
@XmlAccessorType(XmlAccessType.NONE)
public class Author implements Serializable {
    private static final long serialVersionUID = 1L;

    @XmlAttribute
    private String id;

    @XmlElement
    private String firstName;
    @XmlElement
    private String middleName;
    @XmlElement
    private String lastName;
    @XmlElement
    private String email;

    public Author() {}

    public Author(String id) {
        this.id = id;
    }
    
    @Id
    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    @Column(name="first_name", nullable=false)
    public String getFirstName() {
        return firstName;
    }

    public void setFirstName(String firstName) {
        this.firstName = firstName;
    }

    @Column(name="middle_name")
    public String getMiddleName() {
        return middleName;
    }

    public void setMiddleName(String middleName) {
        this.middleName = middleName;
    }

    @Column(name="last_name", nullable=false)
    public String getLastName() {
        return lastName;
    }

    public void setLastName(String lastName) {
        this.lastName = lastName;
    }

    @Transient
    public String getCompleteName() {
        StringBuffer completeName = new StringBuffer();
        completeName.append(this.firstName);
        if(this.middleName != null && !this.middleName.isEmpty()) {
            completeName.append(" ");
            completeName.append(this.middleName);
        }
        completeName.append(" ");
        completeName.append(this.lastName);
        return completeName.toString();
    }

    public String getEmail() {
        return email;
    }

    public void setEmail(String email) {
        this.email = email;
    }

    @Override
    public int hashCode() {
        int hash = 0;
        hash += (id != null ? id.hashCode() : 0);
        return hash;
    }

    @Override
    public boolean equals(Object object) {
        // TODO: Warning - this method won't work in the case the id fields are not set
        if (!(object instanceof Author)) {
            return false;
        }
        Author other = (Author) object;
        if ((this.id == null && other.id != null) || (this.id != null && !this.id.equals(other.id))) {
            return false;
        }
        return true;
    }

    @Override
    public String toString() {
        return getCompleteName();
    }
}