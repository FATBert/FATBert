package br.com.levymoreira.model;

import java.io.Serializable;
import javax.persistence.*;

/**
 * Pojo representando um usuario, configurado com annotation para usar o 
 * hibernate.
 * 
 * @author levy
 * @since 27/11/2011
 */
@SuppressWarnings("serial")
@Entity
@Table(name = "USUARIOS")
public class Usuario implements Serializable {

    @Id
    @GeneratedValue(strategy=GenerationType.IDENTITY)
    private Integer id;
    @Column(name="nome", nullable=false, length=30)
    private String nome;
    @Column(name="senha", nullable=false, length=30)
    private String senha;
    /*@Transient
    private int teste; //nao e persistido!*/
    
    //Constructor defalt
    
    public Usuario(){}
    
    //Constructor

    public Usuario(String nome, String senha){
        this.nome = nome;
        this.senha = senha;
    }
    
    //Generated equals e hashcode (CTRL + I)
    
    @Override
    public boolean equals(Object obj) {
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final Usuario other = (Usuario) obj;
        if (this.id != other.id && (this.id == null || !this.id.equals(other.id))) {
            return false;
        }
        return true;
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 89 * hash + (this.id != null ? this.id.hashCode() : 0);
        return hash;
    }

    // Getters and Setters
    
    public Integer getId() {
        return id;
    }

    public String getNome() {
        return nome;
    }

    public void setNome(String nome) {
        this.nome = nome;
    }

    public String getSenha() {
		return senha;
	}

	public void setSenha(String senha) {
		this.senha = senha;
	}

	@Override
    public String toString() {
        return "Id: " + this.getId() + " - Nome: " + this.getNome();
    }  
    
}
