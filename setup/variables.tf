variable "tenancy_ocid" {}
variable "user_ocid" {}
variable "fingerprint" {}
variable "private_key_path" {}
variable "region" {
  default = "us-phoenix-1"
}

variable "compartment_ocid" {}
variable "ssh_public_key" {}

variable "web_instance_count" {
  default = 2
}

variable "app_instance_count" {
  default = 2
}
