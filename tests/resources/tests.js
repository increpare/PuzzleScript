

var inputVals = {0 : "UP",1: "LEFT",2:"DOWN",3:"RIGHT",undo:"UNDO",restart:"RESTART"};

function testFunction(td) {

}

for (var i=0;i<testdata.length;i++) {
	test(
		testdata[i][0], 
		function(num){
			return function(){
				var td = testdata[num];
				var testcode = td[1][0];
				var testinput=td[1][1];
				var testresult=td[1][2];
				var input="";
				for (var j=0;j<testinput.length;j++) {
					if (j>0) {
						input+=", ";
					}
					input += inputVals[testinput[j]];
				}
				var errormessage = testcode+"\n\n\ninput : "+input;
				ok(runTest(td[1]),errormessage);
			};
		}(i)
	);
}