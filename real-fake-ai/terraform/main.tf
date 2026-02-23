provider "aws" {
  region = var.aws_region
}

resource "aws_security_group" "detector_sg" {
  name        = "real-fake-detector-sg"
  description = "Allow inbound traffic for App and SSH"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "app_server" {
  ami           = "ami-0c7217cdde317cfec" # Ubuntu 22.04 LTS
  instance_type = "t3.medium"             # Needed for TensorFlow RAM requirements
  key_name      = "NaveenBanwala"    # <-- CHANGE THIS

  vpc_security_group_ids = [aws_security_group.detector_sg.id]

  tags = {
    Name = "Real-Fake-Detector-EC2"
  }
}

output "instance_ip" {
  value = aws_instance.app_server.public_ip
}