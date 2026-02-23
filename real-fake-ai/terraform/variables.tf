variable "aws_region" {
  default = "us-east-1"
}

variable "instance_type" {
  default = "t3.medium"
}

variable "key_name" {
  description = "Name of the AWS Key Pair"
  default     = "NaveenBanwala"
}