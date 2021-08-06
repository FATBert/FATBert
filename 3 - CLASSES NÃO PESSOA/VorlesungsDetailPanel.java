/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package gui;

import java.util.HashMap;

public class VorlesungsDetailPanel extends javax.swing.JPanel {
    private int id;
    String select;
    Panel p;

    /**
     * VorlesungsDetailPanel(Panel,int,String)
     * @param p - Panel
     * @param id - der Veranstaltung
     * @param select - auswahl...
     */
    public VorlesungsDetailPanel(Panel p,int id,String select) {
    	this.p=p;
    	this.select=select;
        this.id=id;
        initComponents();
    }

    /**
     * initComponents() Initialisiert die Komponenten.
     */
    private void initComponents() {
        titelLabel = new javax.swing.JLabel();
        detailLabel = new javax.swing.JLabel();
        lehrenderLabel = new javax.swing.JLabel();
        timeLabel = new javax.swing.JLabel();
        pruefenderLabel = new javax.swing.JLabel();
        anOabBtn = new javax.swing.JButton();
        jScrollPane1 = new javax.swing.JScrollPane();
        beschreibungsFeld = new javax.swing.JTextArea();
        setPreferredSize(new java.awt.Dimension(700, 500));
        titelLabel.setText("Vorlesungstitel:");
        detailLabel.setText("Beschreibung:");
        lehrenderLabel.setText("Lehrenden:");
        timeLabel.setText("Zeit:");
        pruefenderLabel.setText("Prüfender:");

        
        switch(id){
            case 0:anOabBtn.setVisible(false);break;
            case 1:anOabBtn.setText("Zu dieser Vorlesung anmelden");break;
            default:anOabBtn.setText("Von dieser Vorlesung abmelden");
        }
        anOabBtn.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                anOabBtnActionPerformed(evt);
            }
        });

        beschreibungsFeld.setColumns(20);
        beschreibungsFeld.setRows(5);
        jScrollPane1.setViewportView(beschreibungsFeld);

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(detailLabel)
                    .addComponent(titelLabel)
                    .addComponent(lehrenderLabel)
                    .addComponent(timeLabel)
                    .addComponent(pruefenderLabel)
                    .addComponent(anOabBtn, javax.swing.GroupLayout.PREFERRED_SIZE, 233, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jScrollPane1, javax.swing.GroupLayout.PREFERRED_SIZE, 443, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap(245, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(titelLabel)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(lehrenderLabel)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(timeLabel)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(pruefenderLabel)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(detailLabel)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jScrollPane1, javax.swing.GroupLayout.DEFAULT_SIZE, 203, Short.MAX_VALUE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                .addComponent(anOabBtn)
                .addGap(105, 105, 105))
        );
    }

    /**
     * anOabBtnActionPerformed(ActionEvent)
     * @param evt
     */
    private void anOabBtnActionPerformed(java.awt.event.ActionEvent evt) {
       if (anOabBtn.getText().equals("Zu dieser Vorlesung anmelden"))
            anOabBtn.setText("Von dieser Vorlesung abmelden");
       else anOabBtn.setText("Zu dieser Vorlesung anmelden");
       if (id==1)p.frame.getController().anmeldenZuVeranstaltung(select);
       else if(id>1)p.frame.getController().abmeldenVonVeranstaltung(select);
    }
    
    /**
     * setPanelData(HashMap)
     * @param data
     */
    public void setPanelData(HashMap data){
    	titelLabel.setText("Titel: "+data.get("titel"));
        lehrenderLabel.setText("Lehrender: "+data.get("lehrender"));
        timeLabel.setText("Zeit: "+data.get("zeit"));
        pruefenderLabel.setText("Prüfender: "+data.get("pruefender"));
        beschreibungsFeld.setText((String)data.get("beschreibung"));
    }
    
    private javax.swing.JButton anOabBtn;
    private javax.swing.JTextArea beschreibungsFeld;
    private javax.swing.JLabel detailLabel;
    private javax.swing.JScrollPane jScrollPane1;
    private javax.swing.JLabel lehrenderLabel;
    private javax.swing.JLabel pruefenderLabel;
    private javax.swing.JLabel timeLabel;
    private javax.swing.JLabel titelLabel;
}
