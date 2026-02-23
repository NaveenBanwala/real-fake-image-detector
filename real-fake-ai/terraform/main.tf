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
  # Updated AMI for ap-south-1 (Mumbai)
  ami           = "ami-0522ab6e1ddcc7055" 
  instance_type = "t3.micro"
  key_name      = "NaveenBanwala"

  vpc_security_group_ids = [aws_security_group.detector_sg.id]

  tags = {
    Name = "Real-Fake-Detector-EC2"
  }
}

output "instance_ip" {
  value = aws_instance.app_server.public_ip
}