package ecms.domain;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;

import javax.persistence.CascadeType;
import javax.persistence.Column;
import javax.persistence.Embedded;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.OneToMany;
import javax.persistence.Table;
import javax.persistence.UniqueConstraint;


@Entity
@Table(name = "TUser", uniqueConstraints = { @UniqueConstraint(columnNames = {"login"})})
public class BaseUser  implements Serializable {
	
	@Id
	@GeneratedValue(strategy=GenerationType.AUTO)
	private Long id;
	
	private static final long serialVersionUID = 1L;

	@Column(name="login", length = 100 ,unique=true, nullable = false )
	private String login ; /*email*/
	
	@Column(name="password", length=100, nullable = false)
	private String password ;
	
	@Column(name="status", length=200 )
	private String status;
	
	@Column(name="firstname", length=200 )
	private String firstName ;
	
	@Column(name="lastname", length=200 )
	private String lastName;
	
	@Embedded 
	private Address address ;
	
	@OneToMany(cascade=CascadeType.ALL, mappedBy = "buyer")
	private Collection<Order> orders;

	@OneToMany(cascade=CascadeType.ALL, mappedBy = "owner")
	private Collection<Cart> carts;

	@OneToMany(cascade = CascadeType.ALL, mappedBy = "storeManager")
	private Collection<Store> stores ;
	
	@Column(name = "role")
	private Role role;
	
	public BaseUser() {
		super();
		orders = new ArrayList<>();
		carts  = new ArrayList<>();
		stores = new ArrayList<>();
	}
	
	public BaseUser( final String nom ,final  String prenom , final String login ,
					 final String mdp,final String status, final Role role, Collection<Store> stores,
					 Collection<Order> orders, Collection<Cart> carts) {
		super();
		this.lastName=nom;
		this.firstName=prenom;
		this.login=login;
		this.password=mdp;
		this.status=status;
		
		this.role = role;
		if(stores != null) {
			this.stores = stores;
		} else {
			this.stores = new ArrayList<>();
		}
		if(orders != null) {
			this.orders = orders;
		} else {
			this.orders = new ArrayList<>();
		}
		if(carts != null) {
			this.carts = carts;
		} else {
			this.carts  = new ArrayList<>();			
		}

		
	}

	/**
	 * @return the id
	 */
	public Long getId() {
		return id;
	}

	/**
	 * @param id the id to set
	 */
	public void setId(Long id) {
		this.id = id;
	}

	/**
	 * @return the login
	 */
	public String getLogin() {
		return login;
	}

	/**
	 * @param login the login to set
	 */
	public void setLogin(String login) {
		this.login = login;
	}

	/**
	 * @return the password
	 */
	public String getPassword() {
		return password;
	}

	/**
	 * @param password the password to set
	 */
	public void setPassword(String password) {
		this.password = password;
	}

	/**
	 * @return the status
	 */
	public String getStatus() {
		return status;
	}

	/**
	 * @param status the status to set
	 */
	public void setStatus(String status) {
		this.status = status;
	}

	/**
	 * @return the firstName
	 */
	public String getFirstName() {
		return firstName;
	}

	/**
	 * @param firstName the firstName to set
	 */
	public void setFirstName(String firstName) {
		this.firstName = firstName;
	}

	/**
	 * @return the lastName
	 */
	public String getLastName() {
		return lastName;
	}

	/**
	 * @param lastName the lastName to set
	 */
	public void setLastName(String lastName) {
		this.lastName = lastName;
	}

	/**
	 * @return the address
	 */
	public Address getAddress() {
		return address;
	}

	/**
	 * @param address the address to set
	 */
	public void setAddress(Address address) {
		this.address = address;
	}

	/**
	 * @return the orders
	 */
	public Collection<Order> getOrders() {
		return orders;
	}

	/**
	 * @param orders the orders to set
	 */
	public void setOrders(Collection<Order> orders) {
		this.orders = orders;
	}

	/**
	 * @return the carts
	 */
	public Collection<Cart> getCarts() {
		return carts;
	}

	/**
	 * @param carts the carts to set
	 */
	public void setCarts(Collection<Cart> carts) {
		this.carts = carts;
	}

	/**
	 * @return the stores
	 */
	public Collection<Store> getStores() {
		return stores;
	}

	/**
	 * @param stores the stores to set
	 */
	public void setStores(Collection<Store> stores) {
		this.stores = stores;
	}

	/**
	 * @return the serialversionuid
	 */
	public static long getSerialversionuid() {
		return serialVersionUID;
	}

	/**
	 * @return the role
	 */
	public Role getRole() {
		return role;
	}

	/**
	 * @param role the role to set
	 */
	public void setRole(Role role) {
		this.role = role;
	}
	 
}
