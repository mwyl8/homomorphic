resource "oci_core_instance" "web_instance" {
  count               = var.web_instance_count
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  compartment_id      = var.compartment_ocid
  shape               = "VM.Standard2.1"
  display_name        = "WebServer-${count.index}"
  
  create_vnic_details {
    subnet_id        = oci_core_subnet.web_subnet.id
    assign_public_ip = true
  }
  
  source_details {
    source_type = "image"
    source_id   = data.oci_core_images.oracle_linux.images[0].id
  }
  
  metadata = {
    ssh_authorized_keys = var.ssh_public_key
  }
}
