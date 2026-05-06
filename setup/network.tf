resource "oci_core_vcn" "vcn" {
  cidr_block     = "10.0.0.0/16"
  compartment_id = var.compartment_ocid
  display_name   = "3TierVCN"
}

resource "oci_core_subnet" "web_subnet" {
  cidr_block        = "10.0.1.0/24"
  compartment_id    = var.compartment_ocid
  vcn_id            = oci_core_vcn.vcn.id
  display_name      = "WebSubnet"
  security_list_ids = [oci_core_security_list.web_security_list.id]
}

resource "oci_core_subnet" "app_subnet" {
  cidr_block        = "10.0.2.0/24"
  compartment_id    = var.compartment_ocid
  vcn_id            = oci_core_vcn.vcn.id
  display_name      = "AppSubnet"
  security_list_ids = [oci_core_security_list.app_security_list.id]
}

resource "oci_core_subnet" "db_subnet" {
  cidr_block        = "10.0.3.0/24"
  compartment_id    = var.compartment_ocid
  vcn_id            = oci_core_vcn.vcn.id
  display_name      = "DBSubnet"
  security_list_ids = [oci_core_security_list.db_security_list.id]
}
